"""Lifecycle-safety tests for CGEventTapRunner — pure mocks, no
Accessibility permission required.

The invariants under test:
- native objects (tap/source) are released exactly once, only by the run
  thread's own reaper, and only after CFRunLoopRun has returned;
- stop() never joins inside the lifecycle lock and never frees anything;
- a stop() timeout leaves cleanup to the thread's reaper (automatic, no
  later stop() call required), and start() is refused until it finished;
- concurrent start/stop and stop-from-callback are safe.
"""

from __future__ import annotations

import threading
import time

import pytest

from wenzi import _cgeventtap as cg
from wenzi._cgeventtap import CGEventTapRunner

_MASK = 1 << 10  # kCGEventKeyDown


def _passthrough(proxy, event_type, event, refcon):
    return event


class _FakeCF:
    """In-memory stand-ins for the CoreFoundation/CoreGraphics calls.

    CFRunLoopRun blocks on a per-thread Event; CFRunLoopStop sets it.
    CFRelease records every released handle.
    """

    def __init__(self, monkeypatch):
        self.released: list[int] = []
        self.create_calls = 0
        self._next_tap = 1000
        self._loops: dict[int, threading.Event] = {}
        self._thread_loop: dict[int, int] = {}
        self._next_loop = 1
        self.fail_create = False
        self.ignore_stop = False

        monkeypatch.setattr(cg, "CGEventTapCreate", self._create_tap)
        monkeypatch.setattr(
            cg, "CFMachPortCreateRunLoopSource", lambda a, tap, o: tap + 1
        )
        monkeypatch.setattr(cg, "CFRunLoopGetCurrent", self._get_current)
        monkeypatch.setattr(cg, "CFRunLoopAddSource", lambda *a: None)
        monkeypatch.setattr(cg, "CGEventTapEnable", lambda *a: None)
        monkeypatch.setattr(cg, "CFRunLoopRun", self._run)
        monkeypatch.setattr(cg, "CFRunLoopStop", self._stop)
        monkeypatch.setattr(cg, "CFRelease", self.released.append)

    def _create_tap(self, *args):
        self.create_calls += 1
        if self.fail_create:
            return 0
        self._next_tap += 10
        return self._next_tap

    def _get_current(self):
        loop_id = self._next_loop
        self._next_loop += 1
        self._loops[loop_id] = threading.Event()
        self._thread_loop[threading.get_ident()] = loop_id
        return loop_id

    def _run(self):
        loop_id = self._thread_loop[threading.get_ident()]
        self._loops[loop_id].wait(10)

    def _stop(self, loop_id):
        if self.ignore_stop:
            return
        event = self._loops.get(loop_id)
        if event is not None:
            event.set()


def _wait_idle(runner: CGEventTapRunner, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while runner._state != "idle" and time.monotonic() < deadline:
        time.sleep(0.01)
    return runner._state == "idle"


class TestRunnerLifecycle:
    def test_start_stop_releases_exactly_once(self, monkeypatch):
        fake = _FakeCF(monkeypatch)
        runner = CGEventTapRunner()
        runner.start(_MASK, _passthrough)
        runner.wait_ready()
        assert runner.running is True

        runner.stop()
        assert _wait_idle(runner)
        # source and tap, each exactly once, released by the reaper
        assert sorted(fake.released) == sorted([runner_tap := fake._next_tap, runner_tap + 1])
        assert runner.tap is None
        assert runner._thread is None

    def test_immediate_stop_after_start(self, monkeypatch):
        """stop() racing setup: the thread must skip the loop, reap its
        own objects once, and end IDLE — repeatedly."""
        fake = _FakeCF(monkeypatch)
        for _ in range(10):
            runner = CGEventTapRunner()
            runner.start(_MASK, _passthrough)
            runner.stop()
            assert _wait_idle(runner)
        # Every created tap has exactly its two releases (tap + source)
        assert len(fake.released) == 2 * fake.create_calls
        assert len(set(fake.released)) == len(fake.released)  # no doubles

    def test_stop_is_idempotent_and_concurrent_safe(self, monkeypatch):
        fake = _FakeCF(monkeypatch)
        runner = CGEventTapRunner()
        runner.start(_MASK, _passthrough)
        runner.wait_ready()

        threads = [
            threading.Thread(target=runner.stop, daemon=True)
            for _ in range(4)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)
        runner.stop()  # extra stop after the fact

        assert _wait_idle(runner)
        assert len(fake.released) == 2  # exactly once each

    def test_stop_from_callback_thread_self_reaps(self, monkeypatch):
        """stop() called on the run-loop thread itself (tap callback):
        disable + loop-stop only; the thread's finally reaps once."""
        fake = _FakeCF(monkeypatch)
        runner = CGEventTapRunner()

        def _self_stopping_run():
            runner.stop()  # simulates a callback invoking stop()

        monkeypatch.setattr(cg, "CFRunLoopRun", _self_stopping_run)
        runner.start(_MASK, _passthrough)

        assert _wait_idle(runner)
        assert len(fake.released) == 2
        assert runner.tap is None

    def test_stop_timeout_then_late_exit_auto_reaps(self, monkeypatch):
        """When the loop refuses to stop, stop() gives up WITHOUT freeing;
        the late-exiting thread reaps automatically, and start() is
        refused until that cleanup finished."""
        fake = _FakeCF(monkeypatch)
        fake.ignore_stop = True  # loop cannot be stopped
        gate = threading.Event()
        monkeypatch.setattr(cg, "CFRunLoopRun", lambda: gate.wait(10))
        monkeypatch.setattr(CGEventTapRunner, "_STOP_JOIN_TIMEOUT", 0.2)

        runner = CGEventTapRunner()
        runner.start(_MASK, _passthrough)
        runner.wait_ready()

        runner.stop()  # times out
        assert fake.released == []  # NEVER freed under a live thread
        assert runner._state == "stopping"

        with pytest.raises(RuntimeError):
            runner.start(_MASK, _passthrough)  # refused during cleanup

        gate.set()  # thread exits late → reaper runs, no stop() needed
        assert _wait_idle(runner)
        assert len(fake.released) == 2

        # After cleanup completed, a fresh start() works again
        gate2 = threading.Event()
        monkeypatch.setattr(cg, "CFRunLoopRun", lambda: gate2.wait(10))
        runner.start(_MASK, _passthrough)  # must not raise
        runner.wait_ready()
        assert runner.running is True
        gate2.set()
        assert _wait_idle(runner)
        assert len(fake.released) == 4  # second generation reaped too

    def test_create_failure_reports_and_goes_idle(self, monkeypatch):
        fake = _FakeCF(monkeypatch)
        fake.fail_create = True
        failed = threading.Event()

        runner = CGEventTapRunner()
        runner.start(_MASK, _passthrough, on_create_failed=failed.set)
        runner.wait_ready()

        assert failed.wait(5)
        assert _wait_idle(runner)
        assert fake.released == []  # nothing was created
        # And the runner is reusable
        fake.fail_create = False
        runner.start(_MASK, _passthrough)
        runner.wait_ready()
        runner.stop()
        assert _wait_idle(runner)

    def test_running_true_while_starting(self, monkeypatch):
        """A STARTING runner must report running — callers must not treat
        it as idle and overwrite the live thread."""
        _FakeCF(monkeypatch)
        gate = threading.Event()
        released_gate = threading.Event()

        def _slow_create(*args):
            gate.set()
            released_gate.wait(5)
            return 4242

        monkeypatch.setattr(cg, "CGEventTapCreate", _slow_create)

        runner = CGEventTapRunner()
        runner.start(_MASK, _passthrough)
        assert gate.wait(5)
        assert runner.running is True  # still STARTING, tap not yet set
        released_gate.set()
        runner.wait_ready()
        runner.stop()
        assert _wait_idle(runner)

    def test_start_while_running_is_refused(self, monkeypatch):
        _FakeCF(monkeypatch)
        runner = CGEventTapRunner()
        runner.start(_MASK, _passthrough)
        runner.wait_ready()

        with pytest.raises(RuntimeError):
            runner.start(_MASK, _passthrough)

        runner.stop()
        assert _wait_idle(runner)


class TestReapingBarrier:
    def test_stop_during_reaping_never_touches_freed_handles(
        self, monkeypatch
    ):
        """Barrier: once the reaper detached the handles and is freeing
        them, a concurrent stop() must not issue a single native
        enable/loop-stop call against them."""
        fake = _FakeCF(monkeypatch)
        monkeypatch.setattr(CGEventTapRunner, "_STOP_JOIN_TIMEOUT", 0.2)

        enable_calls: list = []
        stop_calls: list = []
        monkeypatch.setattr(
            cg, "CGEventTapEnable",
            lambda tap, en: enable_calls.append((tap, en)),
        )
        real_stop = fake._stop
        monkeypatch.setattr(
            cg, "CFRunLoopStop",
            lambda loop: stop_calls.append(loop) or real_stop(loop),
        )

        in_release = threading.Event()
        release_gate = threading.Event()

        def _blocking_release(handle):
            # First release call = reaper phase 2 has begun (handles are
            # already detached under the lock in phase 1)
            in_release.set()
            release_gate.wait(5)
            fake.released.append(handle)

        monkeypatch.setattr(cg, "CFRelease", _blocking_release)

        runner = CGEventTapRunner()
        runner.start(_MASK, _passthrough)
        runner.wait_ready()

        runner.stop()  # loop exits; the thread enters the reaper
        assert in_release.wait(5)
        assert runner._state == "reaping"
        assert runner.tap is None  # phase 1 detached the shared handles

        enable_before = len(enable_calls)
        stop_before = len(stop_calls)

        runner.stop()  # concurrent stop while the reaper frees handles

        # No native call may have touched the freed handles
        assert len(enable_calls) == enable_before
        assert len(stop_calls) == stop_before

        release_gate.set()
        assert _wait_idle(runner)
        assert len(fake.released) == 2


class TestDisableReapLinearization:
    def test_disable_and_reap_are_linearized(self, monkeypatch):
        """Barrier: while an external stop() is inside
        CGEventTapEnable(tap, False), the reaper must not detach or free
        the tap — the disable and the release are strictly ordered."""
        fake = _FakeCF(monkeypatch)
        monkeypatch.setattr(CGEventTapRunner, "_STOP_JOIN_TIMEOUT", 1.0)

        in_disable = threading.Event()
        disable_gate = threading.Event()
        disable_calls: list = []

        def _blocking_disable(tap, enabled):
            disable_calls.append((tap, enabled))
            # Block ONLY the first disable(False) — stop()'s call.  The
            # reaper's own later disable must pass through, otherwise the
            # old (unlocked) implementation would also stall before
            # CFRelease and the released==[] assert would pass vacuously.
            if enabled is False and not in_disable.is_set():
                in_disable.set()
                disable_gate.wait(5)

        monkeypatch.setattr(cg, "CGEventTapEnable", _blocking_disable)

        runner = CGEventTapRunner()
        runner.start(_MASK, _passthrough)
        runner.wait_ready()
        tap_handle = runner.tap

        stopper = threading.Thread(target=runner.stop, daemon=True)
        stopper.start()
        assert in_disable.wait(5)  # stop() is inside the native disable

        # Force the run loop to return so the reaper wants to reclaim NOW
        for event in fake._loops.values():
            event.set()
        time.sleep(0.15)  # reaper is blocked on the lifecycle lock

        # Before the disable returned, the reaper must have neither
        # detached nor freed anything.  On the old (unlocked)
        # implementation it would already have done both by now.
        assert runner.tap == tap_handle
        assert fake.released == []

        disable_gate.set()
        stopper.join(5)
        assert _wait_idle(runner)

        # Exactly one release each, and every native call used the live
        # handle (all disables strictly precede the release)
        assert sorted(fake.released) == sorted([tap_handle, tap_handle + 1])
        assert all(t == tap_handle for t, _ in disable_calls)


class TestSharedTapGenerationIsolation:
    def test_old_generation_callback_is_inert(self, monkeypatch):
        """A superseded SharedHotkeyTap callback must pass events through
        without reading or executing the new tap's bindings."""
        from wenzi.hotkey import SharedHotkeyTap

        tap = SharedHotkeyTap()
        old_runner = object()
        new_runner = object()
        tap._runner = new_runner
        tap._callback = lambda *a: (_ for _ in ()).throw(
            AssertionError("old-generation callback must not dispatch")
        )

        old_cb = tap._make_runner_callback(old_runner)
        sentinel_event = 0xBEEF
        assert old_cb(None, 10, sentinel_event, None) == sentinel_event

        # The current generation still dispatches
        seen = []
        tap._callback = lambda p, et, ev, rc: seen.append(ev) or ev
        current_cb = tap._make_runner_callback(new_runner)
        assert current_cb(None, 10, sentinel_event, None) == sentinel_event
        assert seen == [sentinel_event]
