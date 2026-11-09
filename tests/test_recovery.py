"""Crash-recovery autosave of unsynced layers (job 010)."""

import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtWidgets import QMessageBox

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import companion.app as app_mod
from companion import recovery
from companion.app import MainWindow

SIZE = (120, 90)


@pytest.fixture
def photos(tmp_path):
    out = []
    for name, color in (("A", (180, 190, 200)), ("B", (20, 120, 60))):
        p = str(tmp_path / f"{name}.png")
        Image.new("RGB", SIZE, color).save(p)
        out.append(p)
    return out


@pytest.fixture
def ask(monkeypatch):
    class Ask:
        answer = QMessageBox.StandardButton.Yes
        calls = []
        during = None
    def fake(parent, title, text, *a, **k):
        Ask.calls.append(text)
        if Ask.during:
            during, Ask.during = Ask.during, None
            during()
        return Ask.answer
    monkeypatch.setattr(QMessageBox, "question", fake)
    return Ask


def add_layer(window, box, color):
    mask = Image.new("L", window.base_image.size, 0)
    mask.paste(255, box)
    var = window._get_composite_image()
    var.paste(color, box)
    window.last_used_mask = mask
    window._on_inpainting_finished([var])


def rdir(window):
    return window._recovery_path()


# --- Qt-free module -----------------------------------------------------------------------

def _fake_layer(i, box):
    mask = Image.new("L", SIZE, 0)
    mask.paste(255, box)
    img = Image.new("RGB", SIZE, (i * 40, 10, 10))
    return SimpleNamespace(layer_id=f"id{i}", name=f"Mod {i}", mask=mask, visible=bool(i % 2),
                           engine_mode="firefly", prompt=None, seed=i, detect_subject=False,
                           enable_grain=True, get_active_image=lambda img=img: img)


def test_module_round_trip_and_atomic_overwrite(tmp_path):
    d = recovery.recovery_dir(str(tmp_path), "photo-1")
    assert os.path.basename(d) != "photo-1" and len(os.path.basename(d)) == 40
    layers = [_fake_layer(1, (10, 10, 30, 30)), _fake_layer(2, (50, 40, 70, 60))]
    recovery.save(d, "photo-1", SIZE, layers)
    recovery.save(d, "photo-1", SIZE, layers[:1])  # older files are removed after the replace
    assert sorted(os.listdir(d))[-1] == "manifest.json" and len(os.listdir(d)) == 3
    got = recovery.load(d, "photo-1", SIZE)
    assert len(got) == 1
    e = got[0]
    assert e["box"] == (0, 0, 55, 55)  # 25 px pad, clipped
    assert e["visible"] is True and e["seed"] == 1 and e["enable_grain"] is True
    assert e["patch"].size == (55, 55) and e["mask"].size == (55, 55)
    assert recovery.load(str(tmp_path / "missing"), "k", SIZE) is None


@pytest.mark.parametrize("corrupt", [
    lambda m: "{not json",
    lambda m: {**m, "base_size": [1, 1]},
    lambda m: {**m, "key": "other"},
    lambda m: {**m, "layers": [{**m["layers"][0], "box": [0, 0, 999, 10]}]},
    lambda m: {**m, "layers": [{**m["layers"][0], "visible": "yes"}]},
    lambda m: {**m, "layers": [{**m["layers"][0], "mask": "../manifest.json"}]},
    lambda m: {**m, "layers": [{**m["layers"][0], "box": [0, 0, 20, 20]}]},  # PNG size != box
])
def test_module_rejects_bad_manifest(tmp_path, corrupt):
    d = str(tmp_path / "r")
    recovery.save(d, "k", SIZE, [_fake_layer(1, (10, 10, 30, 30))])
    path = os.path.join(d, "manifest.json")
    with open(path, encoding="utf-8") as f:
        m = json.load(f)
    bad = corrupt(m)
    with open(path, "w", encoding="utf-8") as f:
        f.write(bad if isinstance(bad, str) else json.dumps(bad))
    with pytest.raises(ValueError):
        recovery.load(d, "k", SIZE)


# --- MainWindow integration ---------------------------------------------------------------

def test_recovery_dir_is_outside_tmp_and_ignored(qtbot, photos):
    window = MainWindow(input_path=photos[0])
    qtbot.addWidget(window)
    assert os.path.basename(app_mod._RECOVERY_ROOT) == ".recovery"
    assert ".tmp" not in os.path.relpath(rdir(window), app_mod._RECOVERY_ROOT)
    window.current_photo_id = "LR-42"
    assert rdir(window) == recovery.recovery_dir(app_mod._RECOVERY_ROOT, "LR-42")
    with open(os.path.join(_PROJECT_ROOT, ".gitignore"), encoding="utf-8") as f:
        assert ".recovery/" in f.read().split()


def test_round_trip_restores_layers(qtbot, photos, ask):
    a, _ = photos
    w1 = MainWindow(input_path=a)
    qtbot.addWidget(w1)
    add_layer(w1, (10, 10, 40, 40), (255, 0, 0))
    add_layer(w1, (30, 30, 90, 70), (0, 255, 0))
    w1._toggle_layer_visibility(0)  # autosaves
    assert os.path.isfile(os.path.join(rdir(w1), "manifest.json"))
    expected = np.asarray(w1._recompute_composite_stack(0))
    assert ask.calls == []

    w2 = MainWindow(input_path=a)  # "new session" on the same photo
    qtbot.addWidget(w2)
    assert ask.calls == ["Restore 2 unsaved layers from a previous session?"]
    assert len(w2.modification_layers) == 2
    assert [layer.visible for layer in w2.modification_layers] == [False, True]
    for l1, l2 in zip(w1.modification_layers, w2.modification_layers):
        assert np.array_equal(np.asarray(l1.mask), np.asarray(l2.mask))
        assert (l1.layer_id, l1.name, l1.engine_mode) == (l2.layer_id, l2.name, l2.engine_mode)
    assert np.array_equal(np.asarray(w2.canvas.get_current_image().convert("RGB")), expected)
    assert w2._layers_synced is False
    assert len(w2.layer_card_widgets) == 2


def test_undone_reset_all_is_recoverable(qtbot, photos, ask):
    a, _ = photos
    w1 = MainWindow(input_path=a)
    qtbot.addWidget(w1)
    add_layer(w1, (10, 10, 40, 40), (255, 0, 0))
    add_layer(w1, (30, 30, 90, 70), (0, 255, 0))
    w1._on_reset_all()
    assert not os.path.exists(rdir(w1))  # autosave of the empty stack: nothing to recover
    w1._on_undo()  # autosaves the restored layers
    assert os.path.isfile(os.path.join(rdir(w1), "manifest.json"))
    expected = np.asarray(w1._recompute_composite_stack(0))

    w2 = MainWindow(input_path=a)  # crash, then a new session on the same photo
    qtbot.addWidget(w2)
    assert ask.calls == ["Restore 2 unsaved layers from a previous session?"]
    assert len(w2.modification_layers) == 2
    assert np.array_equal(np.asarray(w2.canvas.get_current_image().convert("RGB")), expected)


def test_open_file_never_uses_previous_lightroom_photo_recovery(qtbot, photos, ask, monkeypatch):
    a, b = photos
    w0 = MainWindow(input_path=a)
    qtbot.addWidget(w0)
    w0.current_photo_id = "lr-1"
    add_layer(w0, (10, 10, 40, 40), (255, 0, 0))  # recovery filed under the LR photo id
    lr_dir = rdir(w0)

    w = MainWindow(input_path=a)
    qtbot.addWidget(w)
    w.current_photo_id = "lr-1"  # the previous Lightroom selection
    ask.calls.clear()
    monkeypatch.setattr(app_mod.QFileDialog, "getOpenFileName", lambda *x, **k: (b, ""))
    w._on_open_file()
    assert ask.calls == [] and w.modification_layers == []  # same size, but not offered on the disk file
    assert w.current_photo_id is None
    add_layer(w, (10, 10, 40, 40), (0, 255, 0))
    assert rdir(w) != lr_dir and os.path.isfile(os.path.join(rdir(w), "manifest.json"))
    assert os.path.isfile(os.path.join(lr_dir, "manifest.json"))


def test_answering_no_deletes_recovery(qtbot, photos, ask):
    a, _ = photos
    w1 = MainWindow(input_path=a)
    qtbot.addWidget(w1)
    add_layer(w1, (10, 10, 40, 40), (255, 0, 0))
    ask.answer = QMessageBox.StandardButton.No
    w2 = MainWindow(input_path=a)
    qtbot.addWidget(w2)
    assert len(ask.calls) == 1 and w2.modification_layers == []
    assert not os.path.exists(rdir(w2))


def test_corrupt_or_wrong_size_manifest_is_deleted_without_prompt(qtbot, photos, ask):
    a, _ = photos
    w = MainWindow(input_path=a)
    qtbot.addWidget(w)
    d = rdir(w)
    os.makedirs(d)
    with open(os.path.join(d, "manifest.json"), "w") as f:
        f.write("garbage")
    w.load_image_file(a)
    assert ask.calls == [] and not os.path.exists(d)

    recovery.save(d, w._recovery_key(), (10, 10), [])
    w.load_image_file(a)
    assert ask.calls == [] and not os.path.exists(d)


def test_discard_yes_deletes_recovery_and_selection_during_prompt_is_deferred(qtbot, photos, ask):
    a, b = photos
    w1 = MainWindow(input_path=a)
    qtbot.addWidget(w1)
    add_layer(w1, (10, 10, 40, 40), (255, 0, 0))

    holder = {}
    ask.during = lambda: holder["w"]._on_live_photo_selected({"path": b, "photo_id": "2"})
    orig_offer = MainWindow._offer_recovery

    def offer(self):
        holder["w"] = self
        orig_offer(self)
        assert self._confirming_switch is False

    MainWindow._offer_recovery = offer
    try:
        w2 = MainWindow(input_path=a)
    finally:
        MainWindow._offer_recovery = orig_offer
    qtbot.addWidget(w2)
    d = rdir(w2)
    assert len(w2.modification_layers) == 1 and os.path.isdir(d)
    assert ask.calls == ["Restore 1 unsaved layer from a previous session?"]  # no stacked prompt
    # The deferred selection then follows the normal 003 rule: discard prompt, Yes deletes recovery
    qtbot.waitUntil(lambda: w2.input_path == b, timeout=3000)
    assert ask.calls[-1] == "Discard 1 unsaved layer?"
    assert not os.path.exists(d)


def test_reselecting_loaded_photo_does_nothing(qtbot, photos, ask):
    a, _ = photos
    w = MainWindow(input_path=a)
    qtbot.addWidget(w)
    add_layer(w, (10, 10, 40, 40), (255, 0, 0))
    w._on_live_photo_selected({"path": a})
    assert ask.calls == [] and len(w.modification_layers) == 1


def test_sync_deletes_recovery_but_failed_sync_and_close_keep_it(qtbot, photos, ask, tmp_path):
    a, _ = photos
    w = MainWindow(input_path=a, output_path=str(tmp_path / "out.tif"))
    qtbot.addWidget(w)
    add_layer(w, (10, 10, 40, 40), (255, 0, 0))
    d = rdir(w)
    w.save_image_to_disk = lambda: None
    w._on_sync_to_lightroom()
    assert os.path.isdir(d)  # failed sync keeps recovery
    del w.save_image_to_disk
    w._on_sync_to_lightroom()
    assert w._layers_synced is True and not os.path.exists(d)

    add_layer(w, (50, 40, 80, 70), (0, 0, 255))
    assert os.path.isdir(d)
    w._cleanup_temp_files()  # what a clean close does to .tmp
    assert os.path.isdir(d)


def test_autosave_timer_and_errors(qtbot, photos, monkeypatch):
    a, _ = photos
    w = MainWindow(input_path=a)
    qtbot.addWidget(w)
    assert w._autosave_timer.isActive()
    assert w._autosave_timer.interval() == app_mod.CONFIG.AUTOSAVE_INTERVAL_MS
    calls = []
    monkeypatch.setattr(recovery, "save", lambda *a_, **k: calls.append(1))
    w._autosave_dirty = True
    w._on_autosave_timer()  # no layers: nothing to write
    assert calls == []
    add_layer(w, (10, 10, 40, 40), (255, 0, 0))
    assert calls == [1]
    w._on_autosave_timer()  # nothing changed
    assert calls == [1]
    w._autosave_dirty = True
    w._layers_synced = True
    w._on_autosave_timer()  # synced
    assert calls == [1]

    def boom(*a_, **k):
        raise OSError("disk full")
    monkeypatch.setattr(recovery, "save", boom)
    w._layers_synced = False
    w._autosave()  # must not raise
    assert "Autosave failed" in w.statusBar().currentMessage()
    assert w._autosave_dirty is True
