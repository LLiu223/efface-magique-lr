"""Model registry (job 013): registry, engine switching, settings, bridge field, Model dropdown."""

import json
import logging
import urllib.error
import urllib.request

import pytest
import torch
from PIL import Image
from PyQt6.QtCore import QSettings, Qt

import companion.model_engine as model_engine
from companion.models import registry, InpaintModel, DEFAULT_MODEL
from companion.models.lama import LamaModel
from companion.model_engine import InpaintingEngine, EngineMode

EVENTS = []      # ("load"|"unload"|"free", name) in call order
RESIDENT = set()  # ids of fake model instances currently loaded


class _FakeBase(InpaintModel):
    def load(self, device, progress_callback=None):
        EVENTS.append(("load", self.name))
        RESIDENT.add(id(self))
        assert len(RESIDENT) == 1, "two models resident at once"

    def unload(self):
        EVENTS.append(("unload", self.name))
        RESIDENT.discard(id(self))

    def inpaint(self, crop, mask, seed, prompt):
        return crop.convert("RGB")


class FakeA(_FakeBase):
    name, label = "fake_a", "Fake A"


class FakeB(_FakeBase):
    name, label = "fake_b", "Fake B"


class FakeUnavailable(_FakeBase):
    name, label = "fake_off", "Fake Off"

    @classmethod
    def is_available(cls):
        return False

    @classmethod
    def unavailable_reason(cls):
        return "Fake Off weights are missing."


@pytest.fixture
def fakes(monkeypatch):
    """Registry with LaMa plus the fake models; restored afterwards."""
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    for cls in (FakeA, FakeB, FakeUnavailable):
        registry.register(cls)
    EVENTS.clear()
    RESIDENT.clear()
    monkeypatch.setattr(model_engine, "free_device_memory", lambda device: EVENTS.append(("free", None)))
    return registry


@pytest.fixture
def ini_settings(monkeypatch, tmp_path):
    import companion.app as app
    path = str(tmp_path / "settings.ini")
    monkeypatch.setattr(app, "_open_settings", lambda: QSettings(path, QSettings.Format.IniFormat))
    return lambda: QSettings(path, QSettings.Format.IniFormat)


def _wait_warmup(window):
    """Let the warm-up thread finish so model switches apply synchronously (job 018 defers them otherwise)."""
    from PyQt6.QtWidgets import QApplication
    window._prewarm_thread.join(60)
    QApplication.processEvents()  # deliver the queued _prewarm_finished


# (a) registry -------------------------------------------------------------

def test_registry_register_get_all_available(fakes):
    assert registry.get("lama") is LamaModel
    assert registry.get("fake_a") is FakeA
    names = [c.name for c in registry.all()]
    assert names[0] == DEFAULT_MODEL and {"fake_a", "fake_b", "fake_off"} <= set(names)
    assert FakeA in registry.available() and FakeUnavailable not in registry.available()


def test_registry_unknown_name_raises_clear_keyerror(fakes):
    with pytest.raises(KeyError) as exc:
        registry.get("nope")
    msg = str(exc.value)
    assert "'nope'" in msg and "lama" in msg and "fake_a" in msg


def test_lama_is_available_does_not_load(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("is_available must not load weights")
    monkeypatch.setattr(LamaModel, "load", boom)
    assert isinstance(LamaModel.is_available(), bool)


# (b) + (c) engine switching ----------------------------------------------

def test_set_model_unloads_and_frees_before_next_load(fakes):
    engine = InpaintingEngine(device=torch.device("cpu"))
    assert engine.model_name == "lama"
    engine.set_model("fake_a")
    engine.load_model()
    EVENTS.clear()

    engine.set_model("fake_b")
    assert EVENTS == [("unload", "fake_a"), ("free", None)]
    assert engine.model_name == "fake_b" and not engine.is_loaded  # still lazy
    engine.load_model()
    assert EVENTS == [("unload", "fake_a"), ("free", None), ("load", "fake_b")]


def test_set_model_unknown_raises_before_unloading(fakes):
    engine = InpaintingEngine(device=torch.device("cpu"))
    engine.set_model("fake_a")
    engine.load_model()
    EVENTS.clear()
    with pytest.raises(KeyError):
        engine.set_model("nope")
    assert EVENTS == [] and engine.model_name == "fake_a" and engine.is_loaded


def test_only_one_model_resident_across_switches(fakes):
    engine = InpaintingEngine(device=torch.device("cpu"))
    img = Image.new("RGB", (64, 64), (100, 100, 100))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (20, 20, 40, 40))
    for name in ("fake_a", "fake_b", "fake_a", "fake_b", "fake_a"):
        engine.set_model(name)
        engine.generate_variations(img, mask, num_variations=1, base_seed=1)  # lazy load
        assert len(RESIDENT) <= 1
    assert [e for e in EVENTS if e[0] == "load"] == [("load", n) for n in ("fake_a", "fake_b", "fake_a", "fake_b", "fake_a")]


def test_fast_mode_error_names_active_model(fakes, monkeypatch):
    def fail(self, device, progress_callback=None):
        raise OSError("weights missing")
    monkeypatch.setattr(FakeA, "load", fail)
    engine = InpaintingEngine(device=torch.device("cpu"), mode=EngineMode.FAST)
    engine.set_model("fake_a")
    img = Image.new("RGB", (64, 64))
    mask = Image.new("L", (64, 64), 0)
    mask.paste(255, (20, 20, 40, 40))
    with pytest.raises(RuntimeError, match=r"Fast mode unavailable: Fake A model \('fake_a'\)"):
        engine.generate_variations(img, mask, num_variations=1)
    engine.set_mode(EngineMode.FIREFLY)  # Generative still falls back to cv2.inpaint
    assert len(engine.generate_variations(img, mask, num_variations=2, base_seed=3)) == 2


# (e) settings ---------------------------------------------------------------

def test_settings_round_trip_and_fallback(qtbot, fakes, ini_settings, caplog):
    from companion.app import MainWindow, MODEL_SETTINGS_KEY
    window = MainWindow()
    qtbot.addWidget(window)
    _wait_warmup(window)
    assert window.engine.model_name == "lama"
    window.combo_model.setCurrentIndex(window.combo_model.findData("fake_b"))
    assert window.engine.model_name == "fake_b"
    window._settings.sync()
    assert ini_settings().value(MODEL_SETTINGS_KEY) == "fake_b"

    restored = MainWindow()
    qtbot.addWidget(restored)
    assert restored.engine.model_name == "fake_b"
    assert restored.combo_model.currentData() == "fake_b"

    for stale in ("gone_model", "fake_off"):  # unknown, unavailable
        s = ini_settings()
        s.setValue(MODEL_SETTINGS_KEY, stale)
        s.sync()
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            w = MainWindow()
            qtbot.addWidget(w)
        assert w.engine.model_name == "lama" and w.combo_model.currentData() == "lama"
        assert any(stale in r.getMessage() and "falling back" in r.getMessage() for r in caplog.records)


# (f) bridge -------------------------------------------------------------------

def _post(port, data):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/select", data=json.dumps(data).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=3.0) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def test_bridge_model_field_accept_absent_reject(qtbot, fakes):
    from companion.live_bridge import LiveBridgeServer
    server = LiveBridgeServer(preferred_port=0)
    port = server.start()
    emitted = []
    server.photo_selected.connect(emitted.append)
    try:
        with qtbot.waitSignal(server.photo_selected, timeout=3000):
            assert _post(port, {"path": "/x.tif", "model": "fake_a"})[0] == 200
        with qtbot.waitSignal(server.photo_selected, timeout=3000):
            assert _post(port, {"path": "/x.tif"})[0] == 200
        assert emitted[0]["model"] == "fake_a" and "model" not in emitted[1]

        for bad, needle in ((7, "string"), ("nope", "Unknown model"), ("fake_off", "not available"),
                            ("../lama", "Unknown model")):
            with pytest.raises(urllib.error.HTTPError) as exc:
                _post(port, {"path": "/x.tif", "model": bad})
            assert exc.value.code == 400
            assert needle in json.loads(exc.value.read().decode("utf-8"))["error"]
        qtbot.wait(100)
        assert len(emitted) == 2  # rejected requests emitted nothing
    finally:
        server.stop()


# (g) UI dropdown ----------------------------------------------------------------

def test_model_dropdown_lists_registry_and_disables_unavailable(qtbot, fakes, ini_settings):
    from companion.app import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    combo = window.combo_model
    assert [combo.itemData(i) for i in range(combo.count())] == [c.name for c in registry.all()]
    assert [combo.itemText(i) for i in range(combo.count())] == [c.label for c in registry.all()]
    i = combo.findData("fake_off")
    assert not (combo.model().item(i).flags() & Qt.ItemFlag.ItemIsEnabled)
    assert combo.itemData(i, Qt.ItemDataRole.ToolTipRole) == "Fake Off weights are missing."
    assert combo.model().item(combo.findData("fake_a")).flags() & Qt.ItemFlag.ItemIsEnabled


# AC8 running-worker rule ------------------------------------------------------------

def test_model_dropdown_locked_with_erase_button(qtbot, fakes, ini_settings, monkeypatch):
    from companion.app import MainWindow
    window = MainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(model_engine.InpaintingWorker, "start", lambda self: None)
    img = Image.new("RGB", (64, 64))
    window._start_inpainting_pipeline(img, Image.new("L", (64, 64), 0))
    assert not window.btn_erase.isEnabled() and not window.combo_model.isEnabled()
    window._on_inpainting_finished([])
    assert window.btn_erase.isEnabled() and window.combo_model.isEnabled()

    window._start_inpainting_pipeline(img, Image.new("L", (64, 64), 0))
    assert not window.combo_model.isEnabled()
    monkeypatch.setattr("companion.app.QMessageBox.critical", lambda *a, **k: None)
    window._on_inpainting_error("boom")
    assert window.btn_erase.isEnabled() and window.combo_model.isEnabled()


def test_bridge_model_change_deferred_until_worker_finishes(qtbot, fakes, ini_settings, tmp_path):
    from companion.app import MainWindow

    class StubWorker:
        def deleteLater(self):
            pass

    photo = str(tmp_path / "a.tif")
    Image.new("RGB", (40, 30), (10, 20, 30)).save(photo)
    window = MainWindow()
    qtbot.addWidget(window)
    _wait_warmup(window)
    window.worker = StubWorker()
    window._on_live_photo_selected({"path": photo, "model": "fake_b"})
    assert window.engine.model_name == "lama"  # held in _pending_select
    window._on_worker_thread_finished()
    assert window.engine.model_name == "fake_b" and window.combo_model.currentData() == "fake_b"
    assert window.input_path == photo


# job 018: model switch hardening ---------------------------------------------------

def test_switch_during_warmup_does_not_block_and_applies_last_pick(qtbot, fakes, ini_settings, monkeypatch):
    import threading
    import time
    from companion.app import MainWindow, MODEL_SETTINGS_KEY
    gate = threading.Event()

    def gated_load(self, device, progress_callback=None):
        _FakeBase.load(self, device, progress_callback)
        gate.wait(10)  # warm-up holds _load_lock until the test opens the gate
    monkeypatch.setattr(FakeA, "load", gated_load)
    s = ini_settings()
    s.setValue(MODEL_SETTINGS_KEY, "fake_a")
    s.sync()

    window = MainWindow()
    qtbot.addWidget(window)
    try:
        qtbot.waitUntil(lambda: ("load", "fake_a") in EVENTS, timeout=3000)
        for name in ("lama", "fake_b"):  # several picks during warm-up: only the last counts
            t0 = time.monotonic()
            window.combo_model.setCurrentIndex(window.combo_model.findData(name))
            assert time.monotonic() - t0 < 0.5
        assert window.engine.model_name == "fake_a"
        assert window.statusBar().currentMessage() == "Switching to Fake B after warm-up..."
        assert not window.btn_erase.isEnabled()
        window._settings.sync()
        assert ini_settings().value(MODEL_SETTINGS_KEY) == "fake_b"
    finally:
        gate.set()

    qtbot.waitUntil(lambda: window.engine.model_name == "fake_b", timeout=3000)
    # first "free" is the startup lama -> fake_a restore; no load of the in-between lama pick
    assert EVENTS == [("free", None), ("load", "fake_a"), ("unload", "fake_a"), ("free", None)]
    assert window.statusBar().currentMessage() == "Inpainting model: Fake B"
    assert window.btn_erase.isEnabled() and not window.engine.is_loaded
    window.engine.load_model()
    assert EVENTS[-1] == ("load", "fake_b")


def _boom_unload(self):
    RESIDENT.discard(id(self))  # the fake really freed; only the call reports failure
    raise RuntimeError("driver lost")


def test_set_model_unload_failure_still_switches(fakes, monkeypatch, caplog):
    engine = InpaintingEngine(device=torch.device("cpu"))
    engine.set_model("fake_a")
    engine.load_model()
    monkeypatch.setattr(FakeA, "unload", _boom_unload)
    EVENTS.clear()
    with caplog.at_level(logging.WARNING, logger="EffaceMagiqueModelEngine"):
        assert engine.set_model("fake_b") is True  # no re-raise
    assert engine._model is None and EVENTS == [("free", None)]
    assert engine._model_cls is FakeB and not engine.is_loaded
    assert "driver lost" in engine.last_unload_error
    assert any("fake_a" in r.getMessage() and r.exc_info for r in caplog.records)
    engine.load_model()
    assert EVENTS[-1] == ("load", "fake_b") and engine._model is not None


def test_unload_failure_shown_in_status_bar(qtbot, fakes, ini_settings, monkeypatch):
    from companion.app import MainWindow
    monkeypatch.setattr(MainWindow, "_prewarm_engine", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    window.engine.set_model("fake_a")
    window.engine.load_model()
    monkeypatch.setattr(FakeA, "unload", _boom_unload)
    window.combo_model.setCurrentIndex(window.combo_model.findData("fake_b"))
    msg = window.statusBar().currentMessage()
    assert msg.startswith("Inpainting model: Fake B") and "Fake A failed: driver lost" in msg
    assert window.engine.model_name == "fake_b" and not window.engine.is_loaded


def test_erase_blocked_while_switch_pending(qtbot, fakes, ini_settings, monkeypatch):
    from companion.app import MainWindow
    monkeypatch.setattr(MainWindow, "_prewarm_engine", lambda self: None)
    started = []
    monkeypatch.setattr(model_engine.InpaintingWorker, "start", lambda self: started.append(self))
    window = MainWindow()
    qtbot.addWidget(window)
    dialogs = []  # without the guard, the no-image check would pop this dialog
    monkeypatch.setattr("companion.app.QMessageBox.information", lambda *a, **k: dialogs.append(a))
    window._pending_model = "fake_b"
    window._on_run_inpainting()  # what Return/Enter call
    assert dialogs == []
    window._start_inpainting_pipeline(Image.new("RGB", (64, 64)), Image.new("L", (64, 64), 0))  # Generate More
    assert started == [] and window.worker is None
    assert window.statusBar().currentMessage() == "Waiting for model switch..."
    window._apply_pending_model()
    assert window.engine.model_name == "fake_b" and window.btn_erase.isEnabled()


def test_deferred_switch_failure_reenables_erase(qtbot, fakes, ini_settings, monkeypatch):
    from companion.app import MainWindow
    monkeypatch.setattr(MainWindow, "_prewarm_engine", lambda self: None)
    window = MainWindow()
    qtbot.addWidget(window)
    window.btn_erase.setEnabled(False)
    window._pending_model = "gone_model"  # stale name: set_model raises KeyError
    window._apply_pending_model()  # must not raise out of the slot
    assert window.btn_erase.isEnabled() and window._pending_model is None
    assert window.statusBar().currentMessage().startswith("Model switch to gone_model failed")
    assert window.engine.model_name == "lama"
