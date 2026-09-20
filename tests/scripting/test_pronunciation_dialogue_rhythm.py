"""Exercise reader guide ownership independently of audio and native UI."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


class FakePanel:
    def __init__(self):
        self.handlers = {}
        self.events = []
        self.close_handler = None

    def show(self):
        pass

    def on(self, name, handler):
        self.handlers[name] = handler

    def on_close(self, handler):
        self.close_handler = handler

    def close(self):
        self.close_handler()

    def send(self, name, data):
        self.events.append((name, data))


@pytest.fixture
def reader(monkeypatch):
    plugin_dir = Path(__file__).resolve().parents[2] / "plugins" / "pronunciation"
    package_name = "pronunciation_dialogue_rhythm_under_test"
    package = ModuleType(package_name)
    package.__path__ = [str(plugin_dir)]
    rhythm = ModuleType(f"{package_name}.rhythm")
    view = ModuleType(f"{package_name}.rhythm_view")
    view.inject_rhythm_assets = lambda html: html
    monkeypatch.setitem(sys.modules, package_name, package)
    monkeypatch.setitem(sys.modules, rhythm.__name__, rhythm)
    monkeypatch.setitem(sys.modules, view.__name__, view)
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.dialogue", plugin_dir / "dialogue.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    panel = FakePanel()
    jobs = []
    wz = SimpleNamespace(
        ui=SimpleNamespace(webview_panel=lambda **kwargs: panel), run=jobs.append
    )
    module.open_dialogue_panel(wz)
    state = SimpleNamespace(module=module, rhythm=rhythm, panel=panel, jobs=jobs)
    yield state
    for job in jobs:
        job.close()
    panel.close()


def guide(text):
    return {
        "text": text,
        "words": text.split(),
        "stress": [{"index": 0, "note": "主要信息"}],
        "pauses": [],
        "summary": "一种自然日常美式读法。",
    }


def request(reader, request_id, text="A: Take it easy.\nB: I will."):
    reader.panel.handlers["request_rhythm"]({"request_id": request_id, "text": text})


def events(reader, name):
    return [data for event, data in reader.panel.events if event == name]


def test_marking_parses_spoken_text_without_generating_audio(reader, monkeypatch):
    calls = []

    async def analyze(text, **kwargs):
        calls.append(text)
        assert kwargs["should_continue"]()
        kwargs["on_progress"](1, 1)
        return guide(text)

    async def unexpected_audio(*args, **kwargs):
        pytest.fail("Marking must not synthesize audio")

    reader.rhythm.analyze_rhythm = analyze
    monkeypatch.setattr(reader.module, "generate_dialogue_audio_bytes", unexpected_audio)
    request(reader, 1)
    asyncio.run(reader.jobs.pop(0))

    assert calls == ["Take it easy.\n\nI will."]
    parsed = events(reader, "rhythm_parsed")[0]
    assert [segment["speaker"] for segment in parsed["segments"]] == ["A", "B"]
    assert events(reader, "rhythm_result")[0]["result"]["words"] == [
        "Take", "it", "easy.", "I", "will."
    ]
    assert events(reader, "rhythm_progress") == [{"done": 1, "total": 1, "request_id": 1}]
    assert not events(reader, "dialogue_parsed")
    assert not events(reader, "dialogue_audio")


def test_new_request_suppresses_old_result_and_progress(reader):
    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()
        current_checks = []

        async def analyze(text, **kwargs):
            if text == "Old words.":
                started.set()
                await release.wait()
                current_checks.append(kwargs["should_continue"]())
                kwargs["on_progress"](1, 1)
            return guide(text)

        reader.rhythm.analyze_rhythm = analyze
        request(reader, 1, "Old words.")
        old_job = asyncio.create_task(reader.jobs.pop(0))
        await started.wait()
        request(reader, 2, "New words.")
        await reader.jobs.pop(0)
        release.set()
        await old_job
        assert current_checks == [False]

    asyncio.run(exercise())
    assert [result["request_id"] for result in events(reader, "rhythm_result")] == [2]
    assert not events(reader, "rhythm_progress")


@pytest.mark.parametrize("failure", [RuntimeError("Old failure"), asyncio.CancelledError()])
def test_edit_cancels_pending_guide_without_showing_stale_errors(reader, failure):
    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        async def analyze(text, **kwargs):
            started.set()
            await release.wait()
            assert not kwargs["should_continue"]()
            raise failure

        reader.rhythm.analyze_rhythm = analyze
        request(reader, 1)
        job = asyncio.create_task(reader.jobs.pop(0))
        await started.wait()
        reader.panel.handlers["invalidate_rhythm"]({"request_id": 1})
        release.set()
        await job

    asyncio.run(exercise())
    assert not events(reader, "rhythm_result")
    assert not events(reader, "rhythm_error")


def test_old_invalidation_cannot_cancel_new_guide(reader):
    calls = []

    async def analyze(text, **kwargs):
        calls.append(text)
        return guide(text)

    reader.rhythm.analyze_rhythm = analyze
    request(reader, 1, "Old words.")
    request(reader, 2, "New words.")
    reader.panel.handlers["invalidate_rhythm"]({"request_id": 1})
    asyncio.run(reader.jobs.pop(0))
    asyncio.run(reader.jobs.pop(0))
    assert calls == ["New words."]
    assert [result["request_id"] for result in events(reader, "rhythm_result")] == [2]


def test_closing_panel_stops_pending_guide(reader):
    async def exercise():
        started = asyncio.Event()
        release = asyncio.Event()

        async def analyze(text, **kwargs):
            started.set()
            await release.wait()
            assert not kwargs["should_continue"]()
            return guide(text)

        reader.rhythm.analyze_rhythm = analyze
        request(reader, 1)
        job = asyncio.create_task(reader.jobs.pop(0))
        await started.wait()
        reader.panel.close()
        release.set()
        await job

    asyncio.run(exercise())
    assert not events(reader, "rhythm_result")


def test_guide_failure_leaves_existing_audio_available_to_save(reader, monkeypatch, tmp_path):
    async def synthesize(*args, **kwargs):
        return b"test-mp3"

    async def analyze(*args, **kwargs):
        raise RuntimeError("Try again")

    reader.rhythm.analyze_rhythm = analyze
    monkeypatch.setattr(reader.module, "generate_dialogue_audio_bytes", synthesize)
    reader.panel.handlers["read_dialogue"]({"request_id": 42, "text": "Take it easy."})
    asyncio.run(reader.jobs.pop(0))
    request(reader, 1, "Take it easy.")
    asyncio.run(reader.jobs.pop(0))

    output_path = tmp_path / "dialogue.mp3"
    picker = SimpleNamespace(
        setTitle_=lambda value: None,
        setCanCreateDirectories_=lambda value: None,
        setAllowedFileTypes_=lambda value: None,
        setNameFieldStringValue_=lambda value: None,
        runModal=lambda: 1,
        URL=lambda: SimpleNamespace(path=lambda: str(output_path)),
    )
    appkit = ModuleType("AppKit")
    appkit.NSModalResponseOK = 1
    appkit.NSSavePanel = SimpleNamespace(savePanel=lambda: picker)
    monkeypatch.setitem(sys.modules, "AppKit", appkit)
    reader.panel.handlers["save_audio"]({"request_id": 42})

    assert output_path.read_bytes() == b"test-mp3"
    assert events(reader, "save_complete")[0]["request_id"] == 42
    assert events(reader, "rhythm_error") == [{"message": "Try again", "request_id": 1}]
    assert len(events(reader, "dialogue_audio")) == 1


def test_invalid_source_reports_guide_error_without_starting_analysis(reader):
    reader.rhythm.analyze_rhythm = None
    request(reader, 1, "   ")
    assert events(reader, "rhythm_error") == [
        {"message": "Dialogue text is empty", "request_id": 1}
    ]
    assert not reader.jobs


def test_queued_main_thread_delivery_rechecks_current_source(reader, monkeypatch):
    queued = []
    pyobjc = ModuleType("PyObjCTools")
    pyobjc.AppHelper = SimpleNamespace(callAfter=queued.append)
    monkeypatch.setitem(sys.modules, "PyObjCTools", pyobjc)

    async def analyze(text, **kwargs):
        kwargs["on_progress"](1, 1)
        return guide(text)

    reader.rhythm.analyze_rhythm = analyze
    request(reader, 1)
    worker = threading.Thread(target=asyncio.run, args=(reader.jobs.pop(0),))
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert len(queued) == 2
    reader.panel.handlers["invalidate_rhythm"]({"request_id": 1})
    for deliver in queued:
        deliver()
    assert not events(reader, "rhythm_progress")
    assert not events(reader, "rhythm_result")
