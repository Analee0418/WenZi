"""Keep sentence rhythm analysis independent from IPA and audio work."""

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
    package_name = "pronunciation_sentence_rhythm_under_test"
    modules = {}
    for name in ("analyze", "tts", "rhythm", "rhythm_view"):
        module = ModuleType(f"{package_name}.{name}")
        modules[name] = module
        monkeypatch.setitem(sys.modules, module.__name__, module)
    local_calls = []
    audio_calls = []

    def local_analysis(text):
        local_calls.append(text)
        return {"words": [{"word": word, "ipa": "/test/"} for word in text.split()]}

    async def tts(text):
        audio_calls.append(text)
        return {"female": "data:audio/mpeg;base64,dGVzdA=="}

    modules["analyze"].analyze_pronunciation_local = local_analysis
    modules["tts"].generate_tts = tts
    modules["rhythm_view"].inject_rhythm_assets = lambda html: html
    spec = importlib.util.spec_from_file_location(
        package_name, plugin_dir / "__init__.py", submodule_search_locations=[str(plugin_dir)]
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, package_name, module)
    spec.loader.exec_module(module)
    panels = []
    jobs = []

    def create_panel(**kwargs):
        panel = FakePanel()
        panels.append(panel)
        return panel

    wz = SimpleNamespace(ui=SimpleNamespace(webview_panel=create_panel), run=jobs.append)
    module._open_panel(wz, "Take it easy.")
    state = SimpleNamespace(
        module=module, rhythm=modules["rhythm"], analyzer=modules["analyze"],
        tts=modules["tts"], panel=panels[0], panels=panels, jobs=jobs, wz=wz,
        local_calls=local_calls, audio_calls=audio_calls,
    )
    yield state
    for job in jobs:
        job.close()
    for panel in panels:
        panel.close()


def guide(text):
    return {
        "text": text, "words": text.split(), "stress": [], "pauses": [],
        "summary": "一种自然日常美式读法。",
    }


def events(panel, name):
    return [data for event, data in panel.events if event == name]


def test_marking_does_not_restart_ipa_or_audio(reader):
    async def analyze(text, **kwargs):
        assert kwargs["should_continue"]()
        kwargs["on_progress"](1, 1)
        return guide(text)

    reader.rhythm.analyze_rhythm = analyze
    asyncio.run(reader.jobs.pop(0))
    reader.panel.handlers["request_rhythm"]({"request_id": 1})
    asyncio.run(reader.jobs.pop(0))

    assert reader.local_calls == ["Take it easy."]
    assert reader.audio_calls == ["Take it easy."]
    assert events(reader.panel, "audio") == [{"female": "data:audio/mpeg;base64,dGVzdA=="}]
    assert events(reader.panel, "rhythm_progress") == [{"done": 1, "total": 1, "request_id": 1}]
    assert events(reader.panel, "rhythm_result") == [
        {"request_id": 1, "result": guide("Take it easy.")}
    ]
    assert not events(reader.panel, "connected_speech")


def test_guide_works_when_ipa_and_tts_fail(reader):
    def bad_ipa(text):
        raise RuntimeError("IPA unavailable")

    async def bad_tts(text):
        raise RuntimeError("TTS unavailable")

    async def analyze(text, **kwargs):
        return guide(text)

    reader.analyzer.analyze_pronunciation_local = bad_ipa
    reader.tts.generate_tts = bad_tts
    reader.rhythm.analyze_rhythm = analyze
    reader.jobs.pop(0).close()
    reader.module._open_panel(reader.wz, "Another sentence.")
    panel = reader.panels[-1]
    asyncio.run(reader.jobs.pop(0))
    panel.handlers["request_rhythm"]({"request_id": 1})
    asyncio.run(reader.jobs.pop(0))

    assert events(panel, "audio_error") == [{"message": "TTS unavailable"}]
    assert events(panel, "rhythm_result") == [
        {"request_id": 1, "result": guide("Another sentence.")}
    ]


@pytest.mark.parametrize("fail", [False, True])
def test_closing_panel_suppresses_late_guide_events(reader, fail):
    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def analyze(text, **kwargs):
            entered.set()
            await release.wait()
            assert not kwargs["should_continue"]()
            kwargs["on_progress"](1, 1)
            if fail:
                raise RuntimeError("Old failure")
            return guide(text)

        reader.rhythm.analyze_rhythm = analyze
        await reader.jobs.pop(0)
        reader.panel.handlers["request_rhythm"]({"request_id": 1})
        job = asyncio.create_task(reader.jobs.pop(0))
        await entered.wait()
        reader.panel.close()
        release.set()
        await job

    asyncio.run(exercise())
    assert not events(reader.panel, "rhythm_progress")
    assert not events(reader.panel, "rhythm_result")
    assert not events(reader.panel, "rhythm_error")


@pytest.mark.parametrize("fail", [False, True])
def test_new_request_owns_progress_and_result(reader, fail):
    async def exercise():
        entered = asyncio.Event()
        release = asyncio.Event()
        calls = []

        async def analyze(text, **kwargs):
            calls.append(text)
            if len(calls) == 1:
                entered.set()
                await release.wait()
                assert not kwargs["should_continue"]()
                kwargs["on_progress"](1, 1)
                if fail:
                    raise RuntimeError("Superseded request failed")
            else:
                kwargs["on_progress"](1, 1)
            return guide(text)

        reader.rhythm.analyze_rhythm = analyze
        await reader.jobs.pop(0)
        reader.panel.handlers["request_rhythm"]({"request_id": 1})
        old_job = asyncio.create_task(reader.jobs.pop(0))
        await entered.wait()
        reader.panel.handlers["request_rhythm"]({"request_id": 2})
        await reader.jobs.pop(0)
        release.set()
        await old_job

    asyncio.run(exercise())
    assert events(reader.panel, "rhythm_progress") == [{"done": 1, "total": 1, "request_id": 2}]
    assert [item["request_id"] for item in events(reader.panel, "rhythm_result")] == [2]
    assert not events(reader.panel, "rhythm_error")


def test_old_panel_close_cannot_release_new_panel(reader):
    reader.module._open_panel(reader.wz, "New sentence.")
    new_panel = reader.panels[-1]
    reader.panel.close()
    assert reader.module._panel_ref[0] is new_panel


def test_queued_main_thread_delivery_rechecks_closed_panel(reader, monkeypatch):
    queued = []
    pyobjc = ModuleType("PyObjCTools")
    pyobjc.AppHelper = SimpleNamespace(callAfter=queued.append)
    monkeypatch.setitem(sys.modules, "PyObjCTools", pyobjc)

    async def analyze(text, **kwargs):
        kwargs["on_progress"](1, 1)
        return guide(text)

    reader.rhythm.analyze_rhythm = analyze
    asyncio.run(reader.jobs.pop(0))
    reader.panel.handlers["request_rhythm"]({"request_id": 1})
    worker = threading.Thread(target=asyncio.run, args=(reader.jobs.pop(0),))
    worker.start()
    worker.join(timeout=1)
    assert not worker.is_alive()
    assert len(queued) == 2
    reader.panel.close()
    for deliver in queued:
        deliver()
    assert not events(reader.panel, "rhythm_progress")
    assert not events(reader.panel, "rhythm_result")
