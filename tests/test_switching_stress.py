"""Seeded random-sequence stress tests for photo switching and output paths (random.Random only, no hypothesis)."""

import os
import random
import sys

import pytest
from PIL import Image
from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from companion.app import MainWindow

SIZE = (80, 60)
KEYS = ("A", "B", "C")
BASE = {"A": (200, 0, 0), "B": (0, 200, 0), "C": (0, 0, 200)}
RESULT = {"A": (255, 120, 10), "B": (10, 255, 120), "C": (120, 10, 255)}  # erase output per source photo
YES, NO = QMessageBox.StandardButton.Yes, QMessageBox.StandardButton.No


class _RunningStub:
    """Stands in for InpaintingWorker: no thread, no model; the test decides when it finishes."""
    def __init__(self):
        self.running = True

    def isRunning(self):
        return self.running

    def wait(self, ms=None):
        return not self.running

    def deleteLater(self):
        pass


@pytest.fixture(autouse=True)
def _no_model_or_quit(monkeypatch):
    monkeypatch.setattr(MainWindow, "_prewarm_engine", lambda self: None)
    monkeypatch.setattr(QApplication, "quit", lambda *a: None)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QMessageBox, "critical", lambda *a, **k: pytest.fail(f"critical dialog: {a[2:]}"))


@pytest.fixture
def photos(tmp_path):
    paths = {}
    for k in KEYS:
        p = str(tmp_path / f"{k}.tif")
        Image.new("RGB", SIZE, BASE[k]).save(p)
        paths[k] = p
    return paths


def _ai_edit(path):
    base, ext = os.path.splitext(path)
    return f"{base}_ai_edit{ext}"


def _snapshot(paths):
    return {p: (os.stat(p).st_mtime_ns, open(p, "rb").read()) for p in paths}


class Harness:
    """One MainWindow per sequence plus a pure-Python model of what it should look like."""

    def __init__(self, qtbot, monkeypatch, photos, tmp_path, rng, init):
        self.rng, self.photos = rng, photos
        self.key_of = {os.path.abspath(p): k for k, p in photos.items()}
        self.missing = str(tmp_path / "missing.tif")
        self.explicit = str(tmp_path / "explicit.tif") if init == "explicit" else None

        # fakes / recorders
        self.prompt_log = []          # (answer_yes, [nested keys]) written by the fake, consumed by the model
        self.active_prompts = 0
        self.max_prompts = 0
        self.depth = self.max_depth = 0
        self.writes = []              # (path, key loaded at save time)
        self.imports = []
        self.dialog_result = None
        monkeypatch.setattr(QMessageBox, "question", self._fake_question)
        monkeypatch.setattr(QFileDialog, "getOpenFileName", lambda *a, **k: (self.dialog_result or "", ""))

        start = photos["A"] if init in ("A", "explicit") else None
        w = self.window = MainWindow(input_path=start, output_path=self.explicit)
        qtbot.addWidget(w)
        w._cleanup_temp_files = lambda: None  # never wipe the repo's .tmp from a test
        w.live_bridge.queue_import = lambda **kw: self.imports.append(kw)
        orig_select, orig_save = w._on_live_photo_selected, w.save_image_to_disk

        def select(payload):
            self.depth += 1
            self.max_depth = max(self.max_depth, self.depth)
            try:
                return orig_select(payload)
            finally:
                self.depth -= 1

        def save():
            path = orig_save()
            if path:
                self.writes.append((path, self.key_of.get(os.path.abspath(w.input_path))))
            return path

        w._on_live_photo_selected, w.save_image_to_disk = select, save

        # model
        self.photo = "A" if start else None
        self.layers, self.synced, self.photo_id = 0, False, None
        self.running_src = None       # key the running erase started on
        self.pending = None           # key of the deferred selection
        self.explicit_active = self.explicit is not None
        self.originals = _snapshot(photos.values())

    # ---- fakes ---------------------------------------------------------
    def _draw_key(self):
        k = self.rng.choice(KEYS + ("current", "bad"))
        if k == "current":
            k = self.key_of.get(os.path.abspath(self.window.input_path or "")) or "bad"
        return k

    def payload(self, k):
        if k == "bad":
            return self.rng.choice([{}, {"path": self.missing, "photo_id": "x"}])
        return {"path": self.photos[k], "photo_id": f"id-{k}", "original_path": f"orig/{k}.dng"}

    def _fake_question(self, *a, **k):
        self.active_prompts += 1
        self.max_prompts = max(self.max_prompts, self.active_prompts)
        try:
            nested = [self._draw_key() for _ in range(self.rng.choice([0, 0, 1, 2]))]
            for nk in nested:  # Lightroom keeps sending selections while the dialog is "open"
                self.window._on_live_photo_selected(self.payload(nk))
            yes = self.rng.random() < 0.5
            self.prompt_log.append((yes, nested))
            return YES if yes else NO
        finally:
            self.active_prompts -= 1

    # ---- model ---------------------------------------------------------
    def _confirm(self):
        if not self.layers or self.synced:
            return True, None
        assert self.prompt_log, "model expected a discard prompt, window did not ask"
        yes, nested = self.prompt_log.pop(0)
        self.yes_this_step |= yes
        valid = [k for k in nested if k != "bad"]
        return yes, (valid[-1] if valid else None)

    def _load(self, k, live):
        if k != self.photo:
            self.explicit_active = False
        self.photo, self.layers, self.synced = k, 0, False
        self.photo_id = f"id-{k}" if live else None

    def _m_select(self, k):
        if k == "bad":
            return
        if self.running_src:
            self.pending = k
            return
        if self.layers and k == self.photo:
            return
        ok, newer = self._confirm()
        if ok:
            self._load(newer or k, live=True)

    # ---- events --------------------------------------------------------
    def ev_select(self):
        k = self._draw_key()
        self.window._on_live_photo_selected(self.payload(k))
        self._m_select(k)
        return f"select:{k}"

    def ev_open(self):
        k = self.rng.choice(KEYS + (None,))
        self.dialog_result = self.photos[k] if k else None
        self.window._on_open_file()
        if not self.running_src:  # Open File is refused while an erase runs
            if k and self._confirm()[0]:
                self._load(k, live=False)
            self.pending = None
        return f"open:{k or 'cancel'}"

    def ev_erase_start(self):
        if self.photo is None or self.running_src:
            return "erase_start:skip"  # UI has no image / Erase button disabled
        self.window.last_used_mask = Image.new("L", SIZE, 255)
        self.window.worker = _RunningStub()
        self.running_src = self.photo
        return f"erase_start:{self.photo}"

    def ev_erase_finish(self):
        if not self.running_src:
            return "erase_finish:skip"
        src, w = self.running_src, self.window
        stub = w.worker
        w._on_inpainting_finished([Image.new("RGB", SIZE, RESULT[src])])
        stub.running = False
        w._on_worker_thread_finished()
        assert self.photo == src, f"erase on {src} finished while {self.photo} is loaded"  # 3f (model)
        self.layers += 1
        self.synced = False
        self.running_src = None
        pending, self.pending = self.pending, None
        if pending:
            self._m_select(pending)
        return f"erase_finish:{src}"

    def ev_sync(self):
        n_writes = len(self.writes)
        self.window._on_sync_to_lightroom()
        if self.photo:
            assert len(self.writes) == n_writes + 1, "sync did not write"
            self.synced = True
            exp = self.explicit if self.explicit_active else _ai_edit(self.photos[self.photo])
            assert os.path.abspath(self.writes[-1][0]) == os.path.abspath(exp), f"wrote {self.writes[-1][0]}, expected {exp}"  # 3b
            imp = self.imports[-1]
            assert imp["export_path"] == self.writes[-1][0]
            assert imp["photo_id"] == self.photo_id
            assert imp["original_path"] == (f"orig/{self.photo}.dng" if self.photo_id else None)
        else:
            assert len(self.writes) == n_writes
        return f"sync:{self.photo}"

    def ev_close(self):
        self.window.closeEvent(QCloseEvent())
        return "close"

    # ---- invariants ----------------------------------------------------
    def check(self, before_layers, synced_before):
        w = self.window
        after_ids = {id(layer) for layer in w.modification_layers}
        lost = [layer for layer in before_layers if id(layer) not in after_ids]
        assert not lost or self.yes_this_step or synced_before or not before_layers, "layers lost without Yes"  # 3a
        assert not self.prompt_log, f"unexpected prompts: {self.prompt_log}"
        assert (self.key_of.get(os.path.abspath(w.input_path)) if w.input_path else None) == self.photo
        assert len(w.modification_layers) == self.layers
        assert w.current_photo_id == self.photo_id
        if self.photo:
            assert w._layers_synced == self.synced
        # 3b: originals untouched, every write is the loaded photo's own output
        assert _snapshot(self.photos.values()) == self.originals, "an original input file was modified"
        for path, key in self.writes:
            assert key is not None
            assert os.path.abspath(path) in {os.path.abspath(_ai_edit(self.photos[key])),
                                             os.path.abspath(self.explicit or _ai_edit(self.photos[key]))}
            if self.explicit and os.path.abspath(path) == os.path.abspath(self.explicit):
                assert key == "A"
        if self.writes:
            path, key = self.writes[-1]
            px = Image.open(path).convert("RGB").getpixel((40, 30))
            assert px in (BASE[key], RESULT[key]), f"{path} holds pixels {px} not from photo {key}"
        # 3c
        assert w._pending_select is None or isinstance(w._pending_select, dict)
        exp_pending = self.photos[self.pending] if self.pending else None
        assert (w._pending_select or {}).get("path") == exp_pending
        assert self.max_depth <= 2, f"_on_live_photo_selected nested {self.max_depth} deep"
        # 3d
        assert self.max_prompts <= 1 and self.active_prompts == 0
        assert w._confirming_switch is False
        # 3e
        if self.running_src:
            assert w.worker is not None and w.worker.isRunning()
        else:
            assert w.worker is None
        # 3f: every layer on screen came from an erase of the loaded photo
        for layer in w.modification_layers:
            assert layer.inpainted_image.getpixel((40, 30)) == RESULT[self.photo], "erase result applied to wrong photo"


EVENTS = [("ev_select", 30), ("ev_open", 15), ("ev_erase_start", 18), ("ev_erase_finish", 18), ("ev_sync", 12)]


@pytest.mark.parametrize("seed", range(24))
def test_random_switching_sequence(qtbot, monkeypatch, photos, tmp_path, seed):
    rng = random.Random(seed)
    init = ("none", "A", "explicit")[seed % 3]
    h = Harness(qtbot, monkeypatch, photos, tmp_path, rng, init)
    names, weights = zip(*EVENTS)
    seq = list(rng.choices(names, weights, k=50)) + ["ev_close"]
    log = [f"init:{init}"]
    for name in seq:
        before, synced_before = list(h.window.modification_layers), h.window._layers_synced
        h.yes_this_step = False
        h.max_depth = h.max_prompts = 0
        try:
            log.append(getattr(h, name)())
            h.check(before, synced_before)
        except AssertionError as e:
            log.append(f"{name} (failed)")
            raise AssertionError(f"seed={seed} events={log}: {e}") from e


def _start_erase(w):
    w.last_used_mask = Image.new("L", SIZE, 255)
    w.worker = _RunningStub()
    return w.worker


def _finish_erase(w, stub, color):
    w._on_inpainting_finished([Image.new("RGB", SIZE, color)])
    stub.running = False
    w._on_worker_thread_finished()


def test_open_file_during_erase_does_not_move_result_to_other_photo(qtbot, monkeypatch, photos, tmp_path):
    h = Harness(qtbot, monkeypatch, photos, tmp_path, random.Random(0), "A")
    w = h.window
    stub = _start_erase(w)
    h.dialog_result = photos["B"]
    w._on_open_file()
    assert w.input_path == photos["A"]
    assert "erase" in w.statusBar().currentMessage().lower()
    _finish_erase(w, stub, RESULT["A"])
    assert w.input_path == photos["A"]
    assert len(w.modification_layers) == 1
    assert w.modification_layers[0].inpainted_image.getpixel((0, 0)) == RESULT["A"]


def test_open_file_clears_lightroom_identity(qtbot, monkeypatch, photos, tmp_path):
    h = Harness(qtbot, monkeypatch, photos, tmp_path, random.Random(0), "none")
    h.window._on_live_photo_selected(h.payload("A"))
    assert h.window.current_photo_id == "id-A"
    h.dialog_result = photos["B"]
    h.window._on_open_file()
    h.window._on_sync_to_lightroom()
    imp = h.imports[-1]
    assert imp["export_path"] == _ai_edit(photos["B"])
    assert imp["photo_id"] is None and imp["original_path"] is None  # never stack B's edit onto A


def test_standalone_open_save_open_save_writes_two_distinct_files(qtbot, monkeypatch, photos, tmp_path):
    a, b = photos["A"], photos["B"]
    h = Harness(qtbot, monkeypatch, photos, tmp_path, random.Random(0), "none")
    w = h.window
    a_bytes = open(a, "rb").read()
    h.dialog_result = a
    w._on_open_file()
    _finish_erase(w, _start_erase(w), RESULT["A"])
    out_a = w.save_image_to_disk()
    out_a_bytes = open(out_a, "rb").read()
    w._layers_synced = True  # saved; switching needs no prompt
    h.dialog_result = b
    w._on_open_file()
    _finish_erase(w, _start_erase(w), RESULT["B"])
    out_b = w.save_image_to_disk()
    assert os.path.basename(out_a) == "A_ai_edit.tif" and os.path.basename(out_b) == "B_ai_edit.tif"
    assert os.path.dirname(out_a) == os.path.dirname(out_b) == str(tmp_path)
    assert open(a, "rb").read() == a_bytes
    assert open(out_a, "rb").read() == out_a_bytes
    assert Image.open(out_b).convert("RGB").getpixel((0, 0)) == RESULT["B"]
