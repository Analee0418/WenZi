"""Coroutine-based recording flow replacing the thread-heavy RecordingController.

The entire hotkey → record → transcribe → enhance → output pipeline is
expressed as a single linear coroutine running on the shared asyncio event
loop.  All business state lives on that single thread, eliminating the need
for locks, events, and busy-token bookkeeping.

External signals (hotkey release, cancel, restart, mode navigation) are
delivered through an :class:`asyncio.Queue` and consumed inline by the
coroutine, which decides how to react based on its current phase.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import threading
import time
from typing import TYPE_CHECKING

from wenzi import async_loop
from wenzi.config import save_config
from wenzi.controllers import fire_scripting_event
from wenzi.input import type_text
from wenzi.input_context import capture_input_context
from wenzi.ui_helpers import get_frontmost_app, reactivate_app

if TYPE_CHECKING:
    from wenzi.app import WenZiApp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Action signal enum — sent from hotkey thread into the asyncio loop
# ---------------------------------------------------------------------------

class Action(enum.Enum):
    RELEASE = "release"
    CANCEL = "cancel"
    CONFIRM_ASR = "confirm_asr"
    RESTART = "restart"
    MODE_PREV = "mode_prev"
    MODE_NEXT = "mode_next"
    PREVIEW_HISTORY = "preview_history"


def _merge_events(
    *events: asyncio.Event,
) -> tuple[asyncio.Event, list[asyncio.Task]]:
    """Return a new Event that is set when any of *events* is set.

    Also returns the background waiter tasks so the caller can cancel
    them when the merged event is no longer needed (avoid task leaks).
    """
    merged = asyncio.Event()

    async def _waiter(ev: asyncio.Event) -> None:
        await ev.wait()
        merged.set()

    tasks = [asyncio.ensure_future(_waiter(ev)) for ev in events]
    return merged, tasks


class _RestartSession(Exception):
    """Sentinel raised inside the coroutine to trigger a recording restart."""
    def __init__(self, key_name: str = "") -> None:
        self.key_name = key_name


# ---------------------------------------------------------------------------
# RecordingFlow — the main coroutine-based recording controller
# ---------------------------------------------------------------------------

class RecordingFlow:
    """Coroutine-based controller for the hotkey → record → output flow."""

    _OUTPUT_RESTORE_ATTEMPTS = 3
    _OUTPUT_RESTORE_RETRY_DELAY = 0.05

    _DELAYED_START_SECS = 0.35
    # A press shorter than this is a tap (cancel/abort): the engine is
    # never touched, so taps reset instantly and cause no system-audio
    # reconfiguration.  A press that survives the grace is a real
    # recording: the mic warms up (gated) during the rest of the sound
    # window so speech is captured from the moment the window ends.
    _TAP_GRACE_SECS = 0.15
    _START_TIMEOUT = 5.0  # seconds to wait for Recorder.start()

    def __init__(self, app: WenZiApp) -> None:
        self._app = app
        self._loop = async_loop.get_loop()
        self._actions: asyncio.Queue[Action] = asyncio.Queue()
        self._current_task: asyncio.Task | None = None
        # True from the loop callback that accepts a press until the
        # session task is created (or the press is rejected).  Guards the
        # pre-session phase against concurrent presses and lets
        # send_action() accept a RELEASE racing with session startup.
        self._press_pending = False
        # Release token for the app-wide exclusive-op slot, held from a
        # successful claim until the session ends.
        self._op_token: object | None = None
        # Mode override state (carried over from RecordingController)
        self._prefer_mode: str | None = None
        self._saved_mode: tuple | None = None
        self._no_preview_override = False
        self._input_context = None
        self._target_app = None  # NSRunningApplication to reactivate before typing
        # Sub-tasks managed within a session
        self._level_task: asyncio.Task | None = None
        self._live_overlay = None
        # Single-flight audio-shutdown task, one per recording session:
        # recorder stop + streaming cleanup run at most once.  Output restore
        # is a separate task so it can overlap streaming/batch transcription.
        self._audio_shutdown_task: asyncio.Task | None = None
        self._output_restore_task: asyncio.Task | None = None
        # In-flight recorder.start() executor future.  Non-None only
        # between launching the start and awaiting it; every path that
        # leaves that window must go through _settle_pending_start() —
        # a dropped future would leave the microphone open.
        self._pending_start: asyncio.Future | None = None

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def is_busy(self) -> bool:
        """True while a press is being handled or a session is in progress."""
        return self._press_pending or (
            self._current_task is not None and not self._current_task.done()
        )

    @property
    def input_context(self):
        """The input context captured at the last hotkey press."""
        return self._input_context

    # ------------------------------------------------------------------
    # Hotkey thread entry points (thread-safe)
    # ------------------------------------------------------------------

    def on_press(self, key_name: str = "") -> None:
        """Called from hotkey thread when the hotkey is pressed.

        The pending flag is set inside the same loop callback that starts
        the press coroutine.  A RELEASE sent right after the press lands
        behind this callback in the loop's FIFO queue, so it can never
        observe an idle flow and be dropped.
        """
        def _start() -> None:
            if self.is_busy:
                return
            self._press_pending = True
            task = self._loop.create_task(self._handle_press(key_name))
            task.add_done_callback(self._log_future_exception)

        self._loop.call_soon_threadsafe(_start)

    @staticmethod
    def _log_future_exception(future: asyncio.Future) -> None:
        if future.cancelled():
            return
        exc = future.exception()
        if exc is not None:
            logger.error("on_press failed: %s", exc, exc_info=exc)

    def send_action(self, action: Action) -> None:
        """Send an action signal into the recording session (thread-safe).

        Actions are dropped at enqueue time while the flow is idle —
        otherwise a stray action queued between sessions would be misread
        by the next one.  This replaces draining the queue on press,
        which could swallow a quick RELEASE racing with session startup.
        """
        def _put() -> None:
            if not self.is_busy:
                return
            self._actions.put_nowait(action)

        self._loop.call_soon_threadsafe(_put)

    # Adapters so MultiHotkeyListener / app.py can use the same callback
    # names as the old RecordingController.

    def on_hotkey_press(self, key_name: str = "") -> None:
        self.on_press(key_name)

    def on_hotkey_release(self, key_name: str = "") -> None:
        self.send_action(Action.RELEASE)

    def on_restart_recording(self) -> None:
        self.send_action(Action.RESTART)

    def on_cancel_recording(self) -> None:
        self.send_action(Action.CANCEL)

    def on_preview_history(self) -> None:
        self.send_action(Action.PREVIEW_HISTORY)

    def on_mode_prev(self) -> None:
        self.send_action(Action.MODE_PREV)

    def on_mode_next(self) -> None:
        self.send_action(Action.MODE_NEXT)

    # ------------------------------------------------------------------
    # Asyncio-thread internal methods
    # ------------------------------------------------------------------

    async def _handle_press(self, key_name: str) -> None:
        if self._current_task is not None and not self._current_task.done():
            return
        session_started = False
        try:
            session_started = await self._do_handle_press(key_name)
        finally:
            # is_busy stays True throughout: when a session was started,
            # _current_task was assigned before pending is cleared here.
            self._press_pending = False
            if not session_started:
                # This press never became a session: release the op slot
                # if it was claimed, and discard any actions queued for it
                # (e.g. its own RELEASE) so they cannot poison the next
                # session.  Token identity makes the release a no-op when
                # the slot belongs to someone else.
                self._app._end_op(self._op_token)
                self._op_token = None
                self._drain_actions()

    async def _do_handle_press(self, key_name: str) -> bool:
        """Handle one accepted press.  Returns True if a session started."""
        app = self._app

        if app._config_degraded:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(app._show_config_error_alert)
            return False

        if not app._voice_input_available:
            from PyObjCTools import AppHelper
            AppHelper.callAfter(self._try_enable_voice_input)
            return False

        # Claim the app-wide op slot before touching any state (mode
        # overrides, context capture) so a refused press has nothing to
        # roll back.
        token = app._try_begin_op("recording")
        if token is None:
            from wenzi.i18n import t

            logger.info(
                "Recording ignored: another operation owns the app "
                "(model switch in progress?)"
            )
            self._show_error_alert(t("alert.recording.busy"))
            return False
        self._op_token = token

        # Capture the frontmost app before any potentially slow AX context
        # lookup so we can reactivate the original target window later.
        ic_level = (
            app._enhancer.input_context_level
            if app._enhancer
            else app._config.get("ai_enhance", {}).get("input_context", "basic")
        )
        self._target_app, self._input_context = await self._loop.run_in_executor(
            None,
            lambda: (
                get_frontmost_app(),
                capture_input_context(ic_level),
            ),
        )

        # Restore previous override before applying a new one
        self._restore_mode()

        # Apply prefer_mode if configured for this hotkey
        self._prefer_mode = None
        hotkey_value = app._config.get("hotkeys", {}).get(key_name)
        if isinstance(hotkey_value, dict):
            prefer_mode = hotkey_value.get("mode")
            if prefer_mode == "no_preview":
                from wenzi.enhance.enhancer import MODE_OFF
                self._no_preview_override = True
                self._prefer_mode = MODE_OFF
                self._apply_prefer_mode(MODE_OFF)
            elif prefer_mode is not None:
                self._prefer_mode = prefer_mode
                self._apply_prefer_mode(prefer_mode)

        logger.info("Hotkey pressed, starting recording session")
        self._current_task = asyncio.create_task(
            self._recording_session(key_name)
        )
        return True

    # ------------------------------------------------------------------
    # The recording session coroutine
    # ------------------------------------------------------------------

    async def _recording_session(
        self, key_name: str, duck_token: object | None = None,
    ) -> None:
        """The full press → delay → record → transcribe → enhance → output flow."""
        from PyObjCTools import AppHelper

        app = self._app
        streaming = False
        restarted = False
        show_preview_history = False
        # Fresh session → fresh single-flight shutdown slot
        self._audio_shutdown_task = None
        self._output_restore_task = None

        try:
            self._fire_scripting_event("recording_start")

            # ① Play start sound + show indicator
            AppHelper.callAfter(app._set_status, "statusbar.status.recording")
            AppHelper.callAfter(app._sound_manager.play, "start")
            if app._sound_manager.enabled:
                AppHelper.callAfter(app._usage_stats.record_sound_feedback)

            initial_dev = (
                app._recorder.last_device_name
                if app._recording_indicator.show_device_name
                else None
            )
            initial_mode = self._get_mode_label()
            AppHelper.callAfter(
                app._recording_indicator.show, initial_dev, initial_mode,
            )

            if app._transcriber.supports_streaming:
                AppHelper.callAfter(self._show_live_overlay, False)

            # ② Sound delay, split in two phases.  Tap grace: the engine
            # is untouched, so a quick tap resets instantly with no
            # engine churn and no system-audio hiccup.  Warm-up: the
            # press is a real recording — the mic starts CONCURRENTLY
            # with the rest of the window, gated so the start sound is
            # never captured, and speech is caught from the first
            # syllable after the window instead of after a full engine
            # spin-up.  Orphan check first: a leftover engine must stop
            # before any new start touches the hardware.
            if app._recorder.is_recording:
                logger.warning(
                    "Recorder unexpectedly active, "
                    "stopping orphaned session"
                )
                await self._loop.run_in_executor(
                    None, app._recorder.stop
                )

            if app._sound_manager.enabled:
                action = await self._wait_action(
                    Action.RELEASE, Action.CANCEL,
                    Action.RESTART, Action.PREVIEW_HISTORY,
                    timeout=self._TAP_GRACE_SECS,
                )
                if action is None:
                    duck_enabled = bool(
                        app._config.get("audio", {}).get(
                            "duck_system_audio", False
                        )
                    )
                    if duck_token is None:
                        duck_token = await self._begin_output_duck()
                    if duck_enabled:
                        action = self._take_queued_action(
                            Action.RELEASE,
                            Action.CANCEL,
                            Action.RESTART,
                            Action.PREVIEW_HISTORY,
                        )
                    if action is None:
                        self._launch_recorder_start(armed=False)
                        action = await self._wait_action(
                            Action.RELEASE, Action.CANCEL,
                            Action.RESTART, Action.PREVIEW_HISTORY,
                            timeout=(
                                self._DELAYED_START_SECS
                                - self._TAP_GRACE_SECS
                            ),
                        )
                    if action is not None:
                        # The orb must vanish at the release, not after
                        # the in-flight start has been settled below.
                        AppHelper.callAfter(app._recording_indicator.hide)
                        if self._pending_start is None:
                            if action != Action.RESTART:
                                await self._restore_output_duck(duck_token)
                        else:
                            await self._settle_pending_start(
                                duck_token,
                                restore_output=action != Action.RESTART,
                            )
                        if action != Action.RESTART:
                            duck_token = None
                if action in (Action.CANCEL, Action.RELEASE):
                    AppHelper.callAfter(self._reset_to_idle)
                    return
                elif action == Action.PREVIEW_HISTORY:
                    show_preview_history = True
                    AppHelper.callAfter(self._reset_to_idle)
                    return
                elif action == Action.RESTART:
                    raise _RestartSession(key_name)

            # ③ Await the warm-up start, or start now (no sound guard)
            start_future = self._pending_start
            if start_future is None:
                duck_enabled = bool(
                    app._config.get("audio", {}).get(
                        "duck_system_audio", False
                    )
                )
                began_duck = duck_token is None and duck_enabled
                if duck_token is None:
                    duck_token = await self._begin_output_duck()
                if began_duck:
                    action = self._take_queued_action(
                        Action.RELEASE,
                        Action.CANCEL,
                        Action.RESTART,
                        Action.PREVIEW_HISTORY,
                    )
                    if action in (Action.CANCEL, Action.RELEASE):
                        AppHelper.callAfter(self._reset_to_idle)
                        return
                    if action == Action.PREVIEW_HISTORY:
                        show_preview_history = True
                        AppHelper.callAfter(self._reset_to_idle)
                        return
                    if action == Action.RESTART:
                        raise _RestartSession(key_name)
                start_future = self._launch_recorder_start(armed=True)
            try:
                dev_name = await asyncio.wait_for(
                    asyncio.shield(start_future),
                    timeout=self._START_TIMEOUT,
                )
            except TimeoutError:
                logger.error(
                    "Recorder.start() timed out after %.0fs, "
                    "aborting session",
                    self._START_TIMEOUT,
                )
                # Clear before tainting: the taint path owns the teardown
                # of the abandoned start, settling it again is pointless.
                self._pending_start = None
                app._recorder.mark_tainted()
                await self._restore_output_duck(duck_token)
                duck_token = None
                AppHelper.callAfter(self._reset_to_idle)
                return
            except Exception:
                # e.g. a concurrent start() in flight, or engine creation
                # blowing up before the recorder could handle it.
                logger.exception("Recorder.start() failed, aborting session")
                self._pending_start = None
                await self._restore_output_duck(duck_token)
                duck_token = None
                AppHelper.callAfter(self._reset_to_idle)
                return
            self._pending_start = None
            if not app._recorder.is_recording:
                # start() reports engine/finalization failures by returning
                # without recording — never show a live recording UI while
                # no engine is actually capturing audio.
                from wenzi.i18n import t
                from wenzi.scripting.api.alert import alert

                logger.error("Recorder did not start, aborting session")
                alert(t("alert.recording.start_failed"), duration=3.0)
                await self._restore_output_duck(duck_token)
                duck_token = None
                AppHelper.callAfter(self._reset_to_idle)
                return
            await self._refresh_output_duck(duck_token)
            # Sound window over and start committed: audio may flow now.
            # A no-op when the session started armed (sound disabled).
            app._recorder.arm()
            if dev_name and app._recording_indicator.show_device_name:
                AppHelper.callAfter(
                    app._recording_indicator.update_device_name, dev_name
                )
            AppHelper.callAfter(app._recording_indicator.set_recording_active)

            # Start streaming transcription if supported
            streaming = self._start_streaming_if_supported()

            # The indicator's EMA must advance on every existing 20 Hz tick.
            # Skip the task entirely when the visual indicator is disabled.
            if app._recording_indicator.enabled:
                self._level_task = asyncio.create_task(self._poll_level())

            # ④ Wait for user action during recording
            max_sec = app._config.get("audio", {}).get(
                "max_recording_seconds", 120
            )
            action = await self._wait_action(
                Action.RELEASE, Action.CANCEL, Action.RESTART,
                Action.PREVIEW_HISTORY,
                timeout=max_sec,
            )

            if action is None:
                # Timeout — treat as release
                logger.warning(
                    "Recording watchdog triggered — auto-stopping "
                    "(possible missed hotkey release)"
                )

            if action == Action.CANCEL:
                await asyncio.shield(
                    self._ensure_audio_shutdown(
                        streaming,
                        cancel=True,
                        duck_token=duck_token,
                    )
                )
                duck_token = None
                self._cancel_subtasks()
                AppHelper.callAfter(self._reset_to_idle)
                return

            if action == Action.RESTART:
                await asyncio.shield(
                    self._ensure_audio_shutdown(
                        streaming,
                        cancel=True,
                        duck_token=duck_token,
                        restore_output=False,
                    )
                )
                raise _RestartSession(key_name)

            if action == Action.PREVIEW_HISTORY:
                await asyncio.shield(
                    self._ensure_audio_shutdown(
                        streaming,
                        cancel=True,
                        duck_token=duck_token,
                    )
                )
                duck_token = None
                self._cancel_subtasks()
                show_preview_history = True
                AppHelper.callAfter(self._reset_to_idle)
                return

            # ⑤ Release (or timeout) — stop recording.  Streaming (if any)
            # finalizes inside the same single-flight shutdown task: with
            # a non-empty wav it stops for the final text, otherwise it
            # cancels.
            # The orb disappears NOW: keeping it up while recorder.stop()
            # blocks (~0.3s of engine teardown) reads as the app lagging
            # behind the key release.
            AppHelper.callAfter(app._recording_indicator.hide)
            self._cancel_subtasks()

            wav_data, stream_text = await asyncio.shield(
                self._ensure_audio_shutdown(
                    streaming,
                    cancel=False,
                    duck_token=duck_token,
                )
            )
            duck_token = None

            # Record audio duration
            audio_duration = 0.0
            if wav_data:
                try:
                    from wenzi.transcription.base import BaseTranscriber
                    audio_duration = BaseTranscriber.wav_duration_seconds(wav_data)
                    app._usage_stats.record_recording_duration(audio_duration)
                except Exception as e:
                    logger.error("Failed to record duration: %s", e)
            app._last_audio_duration = audio_duration
            self._fire_scripting_event(
                "recording_stop", audio_duration=audio_duration
            )

            if not wav_data:
                # (streaming was already cancelled by the shutdown task —
                # an empty wav never finalizes for a result)
                from wenzi.i18n import t
                from wenzi.scripting.api.alert import alert

                alert(t("alert.recording.empty"), duration=2.0)
                AppHelper.callAfter(self._reset_to_idle)
                return

            # ⑥ Transcribe (or defer to preview/direct for background STT)
            effective_preview = app._preview_enabled and not self._no_preview_override
            if effective_preview and not streaming:
                # Non-streaming preview: open preview immediately and
                # let it run STT in the background (asr_text=None).
                await self._route_to_preview(
                    None, audio_duration, wav_data,
                )
                return

            if not effective_preview and not streaming:
                # Non-streaming direct: show overlay immediately and
                # run STT in the background.
                logger.debug("Routing to direct flow with background STT")
                await self._do_direct_flow(
                    None, wav_data, audio_duration
                )
                logger.debug("Direct flow done, session done")
                return

            # All non-streaming paths returned above; only streaming
            # remains — its final text came from the shutdown task.
            text = stream_text
            self._hide_live_overlay()

            logger.debug("Transcription result: %r", text[:100] if text else None)

            if not text or not text.strip():
                AppHelper.callAfter(app._recording_indicator.hide)
                AppHelper.callAfter(
                    app._set_status, "statusbar.status.empty"
                )
                logger.warning("Transcription returned empty text")
                return

            asr_text = text.strip()

            # ⑦ Route to preview or direct flow
            if effective_preview:
                await self._route_to_preview(
                    asr_text, audio_duration, wav_data,
                )
            else:
                logger.debug("Routing to direct flow")
                await self._do_direct_flow(
                    asr_text, wav_data, audio_duration
                )
                logger.debug("Direct flow done, session done")

        except _RestartSession as rs:
            try:
                self._cancel_subtasks()
                self._hide_live_overlay()
                # No drain here: a RELEASE queued right behind the RESTART
                # belongs to the restarted session and must be delivered.
                next_task = asyncio.create_task(
                    self._recording_session(rs.key_name, duck_token)
                )
                self._current_task = next_task
                restarted = True
            except Exception:
                logger.exception("Failed to hand off restarted recording")
                await self._restore_output_duck(duck_token)
                duck_token = None
                AppHelper.callAfter(self._reset_to_idle)
            return
        except asyncio.CancelledError:
            await self._settle_pending_start(duck_token)
            await self._cleanup_session_audio(streaming, duck_token)
            self._cancel_subtasks()
            AppHelper.callAfter(self._reset_to_idle)
        except Exception:
            logger.exception("Recording session failed")
            # Never leave the microphone or a streaming session open
            # behind a reset UI.
            await self._settle_pending_start(duck_token)
            await self._cleanup_session_audio(streaming, duck_token)
            self._cancel_subtasks()
            AppHelper.callAfter(self._reset_to_idle)
        finally:
            # A session may end with actions still queued (a RELEASE right
            # after CANCEL, a start failure before the wait, ...).  Drop
            # them on the loop thread so they cannot leak into the next
            # session — EXCEPT on restart: queued actions (e.g. a RELEASE
            # right behind the RESTART) belong to the restarted session.
            if not restarted:
                restore_scheduled = await asyncio.shield(
                    self._settle_output_restore()
                )
                if not restore_scheduled:
                    await asyncio.shield(
                        self._restore_output_duck(duck_token)
                    )
                app._end_op(self._op_token)
                self._op_token = None
                self._drain_actions()
                if show_preview_history:
                    AppHelper.callAfter(
                        app._preview_controller.on_show_last_preview
                    )

    # ------------------------------------------------------------------
    # Action waiting
    # ------------------------------------------------------------------

    async def _wait_action(
        self, *expected: Action, timeout: float,
    ) -> Action | None:
        """Wait for one of *expected* actions.

        Inline actions (mode navigation) are handled immediately and do not
        interrupt the wait.  Returns ``None`` on timeout.
        """
        deadline = self._loop.time() + timeout
        while True:
            remaining = deadline - self._loop.time()
            if remaining <= 0:
                return None
            try:
                action = await asyncio.wait_for(
                    self._actions.get(), timeout=remaining
                )
            except TimeoutError:
                return None
            if action in expected:
                return action
            self._handle_inline_action(action)

    def _take_queued_action(self, *expected: Action) -> Action | None:
        """Consume an already queued action without yielding the event loop."""
        while True:
            try:
                action = self._actions.get_nowait()
            except asyncio.QueueEmpty:
                return None
            if action in expected:
                return action
            self._handle_inline_action(action)

    def _handle_inline_action(self, action: Action) -> None:
        if action == Action.MODE_PREV:
            self._navigate_mode(-1)
        elif action == Action.MODE_NEXT:
            self._navigate_mode(+1)

    async def _watch_cancel(
        self,
        cancel_event: asyncio.Event,
        confirm_asr_event: asyncio.Event | None = None,
    ) -> None:
        """Monitor the action queue for CANCEL / CONFIRM_ASR.

        Inline actions (mode nav) are handled directly.  Other actions
        (RELEASE, RESTART, PREVIEW_HISTORY) are put back so the caller
        can handle them after the enhancement finishes.
        """
        try:
            while True:
                action = await self._actions.get()
                if action == Action.CANCEL:
                    cancel_event.set()
                    return
                if action == Action.CONFIRM_ASR and confirm_asr_event is not None:
                    confirm_asr_event.set()
                    return
                if action in (Action.MODE_PREV, Action.MODE_NEXT):
                    self._handle_inline_action(action)
                else:
                    # Put back unhandled actions for the caller
                    self._actions.put_nowait(action)
                    return
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # Preview routing helper
    # ------------------------------------------------------------------

    async def _route_to_preview(
        self,
        asr_text: str | None,
        audio_duration: float,
        wav_data: bytes,
    ) -> None:
        """Route to preview panel via executor thread.

        The preview controller blocks on ``result_event.wait()``
        internally, so it must run in an executor, not on the asyncio
        loop.
        """
        app = self._app
        use_enhance = bool(app._enhancer and app._enhancer.is_active)
        logger.debug("Routing to preview flow (asr_text=%s)",
                      "background STT" if asr_text is None else "ready")
        if asr_text is not None:
            self._fire_scripting_event("transcription_done", asr_text=asr_text)
        await self._loop.run_in_executor(
            None,
            lambda: app._do_transcribe_with_preview(
                asr_text=asr_text,
                use_enhance=use_enhance,
                audio_duration=audio_duration,
                wav_data=wav_data,
            ),
        )
        logger.debug("Preview flow done, session done")

    # ------------------------------------------------------------------
    # Direct output flow (no preview panel)
    # ------------------------------------------------------------------

    async def _do_direct_flow(
        self,
        asr_text: str | None,
        wav_data: bytes,
        audio_duration: float,
    ) -> None:
        from PyObjCTools import AppHelper

        app = self._app
        use_enhance = bool(app._enhancer and app._enhancer.is_active)
        need_stt = asr_text is None

        try:
            app._usage_stats.record_transcription(
                mode="direct", enhance_mode=app._enhance_mode
            )
        except Exception as e:
            logger.error("Failed to record usage stats: %s", e)

        text = asr_text or ""
        enhanced_text = None
        enhance_fell_back = False
        cancel_event = asyncio.Event()

        if use_enhance:
            initial_status = (
                "statusbar.status.transcribing" if need_stt
                else "statusbar.status.enhancing"
            )
            AppHelper.callAfter(app._set_status, initial_status)

            # Enter → output ASR directly, ESC → discard all.
            confirm_asr_event = asyncio.Event()
            cancel_watcher = asyncio.create_task(
                self._watch_cancel(cancel_event, confirm_asr_event)
            )

            stt_info = app._current_stt_model()
            llm_info = app._current_llm_model()

            AppHelper.callAfter(
                self._show_streaming_overlay,
                asr_text or "", stt_info, llm_info, True,
            )

            # Background STT if needed
            if need_stt:
                asr_text = await self._background_stt(
                    wav_data, cancel_event, skip_punc=True
                )
                if cancel_event.is_set():
                    cancel_watcher.cancel()
                    AppHelper.callAfter(app._streaming_overlay.close)
                    AppHelper.callAfter(
                        app._set_status, "statusbar.status.ready"
                    )
                    return
                if not asr_text:
                    cancel_watcher.cancel()
                    AppHelper.callAfter(app._streaming_overlay.close)
                    AppHelper.callAfter(
                        app._set_status, "statusbar.status.empty"
                    )
                    return
                text = asr_text

                # Enter pressed during STT: skip enhancement, output ASR
                if confirm_asr_event.is_set():
                    logger.debug("Enter during STT: skipping enhance")
                    cancel_watcher.cancel()
                    AppHelper.callAfter(
                        app._streaming_overlay.close_now
                    )
                    self._fire_scripting_event(
                        "transcription_done", asr_text=asr_text
                    )
                else:
                    AppHelper.callAfter(
                        app._set_status, "statusbar.status.enhancing"
                    )

            if not confirm_asr_event.is_set():
                self._fire_scripting_event(
                    "transcription_done", asr_text=asr_text
                )

                # Resolve chain steps
                try:
                    current_mode_def = app._enhancer.get_mode_definition(
                        app._enhance_mode
                    )
                    chain_steps: list[str] = []
                    if current_mode_def and current_mode_def.steps:
                        for step_id in current_mode_def.steps:
                            step_def = app._enhancer.get_mode_definition(
                                step_id
                            )
                            if step_def:
                                chain_steps.append(step_id)
                            else:
                                logger.warning(
                                    "Chain step '%s' not found, skipping",
                                    step_id,
                                )

                    # Both cancel_event and confirm_asr_event abort the stream
                    abort_event, abort_tasks = _merge_events(
                        cancel_event, confirm_asr_event
                    )

                    try:
                        if chain_steps:
                            text, enhance_fell_back = (
                                await self._run_direct_chain_stream(
                                    asr_text, chain_steps, abort_event
                                )
                            )
                        else:
                            text, enhance_fell_back = (
                                await self._run_direct_single_stream(
                                    asr_text, abort_event
                                )
                            )
                    finally:
                        for t in abort_tasks:
                            t.cancel()
                        results = await asyncio.gather(
                            *abort_tasks, return_exceptions=True
                        )
                        for r in results:
                            if isinstance(r, Exception) and not isinstance(
                                r, asyncio.CancelledError
                            ):
                                logger.warning(
                                    "abort waiter error: %s", r
                                )

                    if confirm_asr_event.is_set():
                        logger.debug("Enter during enhance: using ASR text")
                        text = asr_text
                        enhanced_text = None
                    elif cancel_event.is_set():
                        text = asr_text
                        enhanced_text = None
                    elif enhance_fell_back:
                        # Fallback output is the original text, not an
                        # enhancement result: don't fire enhancement_done
                        # or log it into history as enhanced.
                        enhanced_text = None
                    else:
                        enhanced_text = text
                        self._fire_scripting_event(
                            "enhancement_done", enhanced_text=enhanced_text
                        )
                except Exception as e:
                    logger.error("AI enhancement failed: %s", e)
                    text = asr_text
                finally:
                    cancel_watcher.cancel()
                    if cancel_event.is_set() or confirm_asr_event.is_set():
                        # Use _do_close directly so the panel is removed
                        # before type_text runs in the same callAfter queue.
                        AppHelper.callAfter(
                            app._streaming_overlay.close_now
                        )
                    else:
                        AppHelper.callAfter(
                            app._streaming_overlay.close_with_delay
                        )
        else:
            # No enhancement — show overlay for background STT if needed.
            # Streaming already has ASR text; just hide the indicator.
            if need_stt:
                cancel_watcher = asyncio.create_task(
                    self._watch_cancel(cancel_event)
                )
                AppHelper.callAfter(
                    self._show_streaming_overlay,
                    "", app._current_stt_model(), "",
                )

                asr_text = await self._background_stt(
                    wav_data, cancel_event
                )
                cancel_watcher.cancel()
                if cancel_event.is_set() or not asr_text:
                    AppHelper.callAfter(app._streaming_overlay.close)
                    AppHelper.callAfter(
                        app._set_status,
                        "statusbar.status.ready"
                        if cancel_event.is_set()
                        else "statusbar.status.empty",
                    )
                    return
                text = asr_text
                AppHelper.callAfter(
                    app._streaming_overlay.close_with_delay
                )
            else:
                AppHelper.callAfter(app._recording_indicator.hide)

            self._fire_scripting_event(
                "transcription_done", asr_text=asr_text
            )

        logger.debug(
            "Direct flow output: cancel=%s, confirm_asr=%s, text=%r",
            cancel_event.is_set(),
            confirm_asr_event.is_set() if use_enhance else "N/A",
            text[:50] if text else None,
        )

        if cancel_event.is_set():
            AppHelper.callAfter(
                app._set_status, "statusbar.status.ready"
            )
            return

        text = text.strip()

        self._fire_scripting_event(
            "output_text", final_text=text
        )

        # Reactivate the app that was frontmost when the hotkey was pressed,
        # in case the user switched windows during AI enhancement.
        if self._target_app is not None:
            AppHelper.callAfter(reactivate_app, self._target_app)
            await asyncio.sleep(0.15)

        AppHelper.callAfter(
            type_text,
            text,
            append_newline=app._append_newline,
            method=app._output_method,
        )
        AppHelper.callAfter(app._set_status, "statusbar.status.ready")

        try:
            app._usage_stats.record_confirm(modified=False)
        except Exception as e:
            logger.error("Failed to record usage stats: %s", e)
        try:
            app._usage_stats.record_output_method(copy_to_clipboard=False)
        except Exception as e:
            logger.error("Failed to record output method: %s", e)

        self._target_app = None

        try:
            app._conversation_history.log(
                asr_text=asr_text,
                enhanced_text=enhanced_text,
                final_text=text,
                enhance_mode=app._enhance_mode,
                preview_enabled=False,
                stt_model=app._current_stt_model(),
                llm_model=app._current_llm_model(),
                audio_duration=getattr(app, "_last_audio_duration", 0.0),
                input_context=self._input_context,
            )
        except Exception as e:
            logger.error("Failed to log conversation: %s", e)

    # ------------------------------------------------------------------
    # Overlay helpers for direct flow
    # ------------------------------------------------------------------

    def _show_streaming_overlay(
        self,
        asr_text: str,
        stt_info: str,
        llm_info: str,
        with_confirm_asr: bool = False,
    ) -> None:
        """Hide recording indicator and show the streaming overlay immediately.

        Must be called on the main thread (via AppHelper.callAfter).
        """
        app = self._app
        indicator_frame = app._recording_indicator.current_frame
        app._recording_indicator.hide()
        app._streaming_overlay.show(
            asr_text=asr_text,
            animate_from_frame=indicator_frame,
            stt_info=stt_info,
            llm_info=llm_info,
            on_cancel=lambda: self.send_action(Action.CANCEL),
            on_confirm_asr=(
                lambda: self.send_action(Action.CONFIRM_ASR)
            ) if with_confirm_asr else None,
        )

    @staticmethod
    def _show_error_alert(message: str, duration: float = 3.0) -> None:
        """Show a lightweight floating alert for errors."""
        try:
            from wenzi.scripting.api.alert import alert
            alert(message, duration=duration)
        except Exception:
            logger.debug("Error alert failed", exc_info=True)

    def _apply_timeout_fallback(
        self, app, chunk: str, step_idx: int = 0, total_steps: int = 0,
    ) -> tuple[list[str], int]:
        """Update overlay with fallback text on AI timeout."""
        completion_tokens = len(chunk)
        app._streaming_overlay.clear_text()
        app._streaming_overlay.append_text(chunk, completion_tokens=completion_tokens)
        if total_steps > 0:
            msg = (
                f"\u26a0\ufe0f Step {step_idx}/{total_steps}: "
                "AI enhancement failed, using original text"
            )
        else:
            msg = "\u26a0\ufe0f AI enhancement failed, using original text"
        app._streaming_overlay.set_status(msg)
        self._show_error_alert("AI enhancement failed, original text used")
        return [chunk], completion_tokens

    # ------------------------------------------------------------------
    # Background STT for direct flow
    # ------------------------------------------------------------------

    async def _background_stt(
        self,
        wav_data: bytes,
        cancel_event: asyncio.Event,
        skip_punc: bool = False,
    ) -> str | None:
        """Run STT in background, update overlay when done.

        Returns the transcribed text (stripped), or None on empty/failure.
        """
        from PyObjCTools import AppHelper

        app = self._app
        app._transcriber.skip_punc = skip_punc
        hotwords, _ = app._build_dynamic_hotwords()

        text = await self._loop.run_in_executor(
            None, lambda: app._transcriber.transcribe(
                wav_data, hotwords=hotwords
            )
        )

        if cancel_event.is_set():
            return None

        asr_text = (text or "").strip()
        logger.debug("Background STT result: %r", asr_text[:100] if asr_text else None)

        if asr_text:
            AppHelper.callAfter(
                app._streaming_overlay.set_asr_text, asr_text
            )

        return asr_text or None

    # ------------------------------------------------------------------
    # Streaming enhancement helpers (native async — no event loop hacks)
    # ------------------------------------------------------------------

    @staticmethod
    async def _iter_or_cancel(gen, cancel_event: asyncio.Event):
        """Iterate an async generator, aborting immediately on cancel.

        Races each ``__anext__`` against ``cancel_event.wait()`` so
        cancellation is detected even while blocked waiting for a chunk.
        The generator is always closed (``aclose``) on exit.
        """
        aiter = gen.__aiter__()
        cancel_fut = asyncio.ensure_future(cancel_event.wait())
        try:
            while True:
                next_fut = asyncio.ensure_future(aiter.__anext__())
                done, _ = await asyncio.wait(
                    [next_fut, cancel_fut],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if cancel_fut in done:
                    next_fut.cancel()
                    # Wait for __anext__ to finish cancellation before
                    # closing the generator (avoids "already running").
                    try:
                        await next_fut
                    except (asyncio.CancelledError, StopAsyncIteration):
                        pass
                    return
                try:
                    yield next_fut.result()
                except StopAsyncIteration:
                    return
        finally:
            cancel_fut.cancel()
            await gen.aclose()

    async def _run_direct_single_stream(
        self, asr_text: str, cancel_event: asyncio.Event,
    ) -> tuple[str, bool]:
        """Single-step streaming enhancement, updating overlay.

        Returns ``(text, fell_back)`` — *fell_back* is True when the
        stream failed and *text* is the original-text fallback.
        """
        app = self._app
        collected: list[str] = []
        usage = None
        completion_tokens = 0
        thinking_tokens = 0
        had_thinking = False
        fell_back = False

        gen = app._enhancer.enhance_stream(
            asr_text, input_context=self._input_context
        )
        async for chunk, chunk_usage, is_thinking in self._iter_or_cancel(
            gen, cancel_event
        ):
            if is_thinking == "retry" and chunk:
                had_thinking = True
                app._streaming_overlay.append_thinking_text(chunk)
                label = chunk.strip().strip("()\n")
                app._streaming_overlay.set_status(f"\u23f3 {label}")
            elif is_thinking == "timeout" and chunk:
                had_thinking = False
                collected, completion_tokens = self._apply_timeout_fallback(
                    app, chunk,
                )
                fell_back = True
                break
            elif is_thinking and chunk:
                had_thinking = True
                thinking_tokens += len(chunk)
                app._streaming_overlay.append_thinking_text(
                    chunk, thinking_tokens=thinking_tokens
                )
            elif chunk:
                if had_thinking:
                    had_thinking = False
                    app._streaming_overlay.clear_text()
                collected.append(chunk)
                completion_tokens += len(chunk)
                app._streaming_overlay.append_text(
                    chunk, completion_tokens=completion_tokens
                )
            if chunk_usage is not None:
                usage = chunk_usage

        if usage:
            try:
                app._usage_stats.record_token_usage(usage)
            except Exception as e:
                logger.error("Failed to record token usage: %s", e)
            if not fell_back:
                app._streaming_overlay.set_complete(usage)

        return "".join(collected).strip() or asr_text, fell_back

    async def _run_direct_chain_stream(
        self,
        asr_text: str,
        chain_steps: list[str],
        cancel_event: asyncio.Event,
    ) -> tuple[str, bool]:
        """Multi-step chain streaming enhancement, updating overlay.

        Returns ``(text, fell_back)`` — on a failed step the chain aborts
        and falls back to the original ASR text (displayed AND returned,
        so the overlay never shows something different from what gets
        typed).
        """
        app = self._app
        total_steps = len(chain_steps)
        input_text = asr_text
        original_mode = app._enhancer.mode
        chain_fell_back = False
        total_usage: dict[str, int] = {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        }

        try:
            for step_idx, step_id in enumerate(chain_steps, 1):
                if cancel_event.is_set():
                    break

                step_def = app._enhancer.get_mode_definition(step_id)
                step_label = step_def.label if step_def else step_id

                app._streaming_overlay.set_status(
                    f"\u23f3 Step {step_idx}/{total_steps}: {step_label}"
                )
                app._streaming_overlay.set_progress(step_idx, total_steps)

                if step_idx > 1:
                    app._streaming_overlay.clear_text()

                app._enhancer.mode = step_id

                collected: list[str] = []
                step_usage = None
                completion_tokens = 0
                thinking_tokens = 0

                gen = app._enhancer.enhance_stream(
                    input_text, input_context=self._input_context
                )
                async for chunk, chunk_usage, is_thinking in self._iter_or_cancel(
                    gen, cancel_event
                ):
                    if is_thinking == "retry" and chunk:
                        app._streaming_overlay.append_thinking_text(chunk)
                        label = chunk.strip().strip("()\n")
                        app._streaming_overlay.set_status(
                            f"\u23f3 Step {step_idx}/{total_steps}: {label}"
                        )
                    elif is_thinking == "timeout" and chunk:
                        # Display the ORIGINAL text as the fallback — the
                        # chain aborts and returns asr_text, and the
                        # overlay must match what will be typed (the
                        # step's input would show a half-chain result).
                        collected, completion_tokens = self._apply_timeout_fallback(
                            app, asr_text, step_idx, total_steps,
                        )
                        chain_fell_back = True
                        break
                    elif is_thinking and chunk:
                        thinking_tokens += len(chunk)
                        app._streaming_overlay.append_thinking_text(
                            chunk, thinking_tokens=thinking_tokens
                        )
                    elif chunk:
                        collected.append(chunk)
                        completion_tokens += len(chunk)
                        app._streaming_overlay.append_text(
                            chunk, completion_tokens=completion_tokens
                        )
                    if chunk_usage is not None:
                        step_usage = chunk_usage

                if chain_fell_back:
                    # Abort the chain: later steps would run on stale
                    # input and burn tokens for a result we won't use.
                    break

                step_result = "".join(collected).strip()
                if step_result:
                    input_text = step_result

                if step_usage:
                    total_usage["prompt_tokens"] += step_usage.get(
                        "prompt_tokens", 0
                    )
                    total_usage["completion_tokens"] += step_usage.get(
                        "completion_tokens", 0
                    )
                    total_usage["total_tokens"] += step_usage.get(
                        "total_tokens", 0
                    )
                try:
                    app._usage_stats.record_token_usage(step_usage)
                except Exception as e:
                    logger.error("Failed to record token usage: %s", e)

            if chain_fell_back:
                # Aborted chain falls back to the original text; don't
                # mark the run complete — the overlay already shows the
                # fallback text and the failure status.
                return asr_text, True

            if total_usage["total_tokens"] > 0:
                app._streaming_overlay.set_complete(total_usage)

            return input_text.strip() or asr_text, False
        finally:
            app._enhancer.mode = original_mode

    # ------------------------------------------------------------------
    # Streaming transcription
    # ------------------------------------------------------------------

    def _start_streaming_if_supported(self) -> bool:
        """Start streaming transcription synchronously. Returns success."""
        from PyObjCTools import AppHelper

        app = self._app
        if not app._transcriber.supports_streaming:
            return False
        try:
            def _on_partial(text: str, is_final: bool) -> None:
                AppHelper.callAfter(self._update_live_overlay, text)

            app._transcriber.start_streaming(_on_partial)
            try:
                app._recorder.set_on_audio_chunk(app._transcriber.feed_audio)
            except Exception:
                # Half-started: never leave a background recognizer running
                logger.exception(
                    "Audio-chunk attach failed; cancelling streaming"
                )
                try:
                    app._transcriber.cancel_streaming()
                except Exception:
                    logger.exception("Cancel after attach failure failed")
                return False

            # Activate the overlay (already shown in faded state)
            if self._live_overlay is not None:
                AppHelper.callAfter(self._live_overlay.set_active)
            else:
                AppHelper.callAfter(self._show_live_overlay)
            logger.info("Streaming transcription started")
            return True
        except Exception:
            logger.exception("Failed to start streaming, will use batch mode")
            # start_streaming may have allocated backend resources before
            # raising — best-effort cancel so no half-started recognizer
            # session leaks.
            try:
                app._transcriber.cancel_streaming()
            except Exception:
                logger.exception("Cancel after failed start also failed")
            return False

    async def _begin_output_duck(self) -> object | None:
        """Lower system playback before touching the microphone hardware."""
        audio_cfg = self._app._config.get("audio", {})
        if not audio_cfg.get("duck_system_audio", False):
            return None
        try:
            factor = float(audio_cfg.get("duck_volume_ratio", 0.25))
            max_volume = float(audio_cfg.get("duck_max_volume", 0.05))
            begin_future = asyncio.ensure_future(
                self._loop.run_in_executor(
                    None,
                    lambda: self._app._system_output_ducker.begin(
                        factor=factor,
                        max_volume=max_volume,
                    ),
                )
            )
            try:
                return await asyncio.shield(begin_future)
            except asyncio.CancelledError:
                # Cancelling an asyncio waiter cannot stop its executor
                # job. Settle it here so a token returned after cancellation
                # is never lost with the system volume left lowered.
                token = await asyncio.shield(begin_future)
                if token is not None:
                    try:
                        await asyncio.shield(
                            self._loop.run_in_executor(
                                None,
                                lambda: self._end_output_duck_sync(token),
                            )
                        )
                    except Exception:
                        logger.exception(
                            "Failed to restore output after cancelled start"
                        )
                raise
        except Exception:
            # Recording remains usable when an output device exposes no
            # writable volume control or CoreAudio rejects the request.
            logger.exception("Failed to lower system output volume")
            return None

    async def _refresh_output_duck(self, duck_token: object | None) -> None:
        """Apply the same cap if microphone startup changed the output route."""
        if duck_token is None:
            return
        try:
            await self._loop.run_in_executor(
                None,
                lambda: self._app._system_output_ducker.refresh(duck_token),
            )
        except Exception:
            logger.exception("Failed to lower the updated output route")

    async def _restore_output_duck(self, duck_token: object | None) -> None:
        """Best-effort, idempotent playback-volume restore."""
        if duck_token is None:
            return
        try:
            await self._loop.run_in_executor(
                None,
                lambda: self._end_output_duck_sync(duck_token),
            )
        except Exception:
            logger.exception("Failed to restore system output volume")

    def _ensure_audio_shutdown(
        self,
        streaming: bool,
        cancel: bool,
        *,
        duck_token: object | None = None,
        restore_output: bool = True,
    ) -> asyncio.Task:
        """Return this session's single-flight audio-shutdown task.

        The first caller creates it; every later caller — including the
        exception and cancel handlers — awaits the SAME task, so the
        recorder stop and the streaming cleanup run at most once on one
        executor job.  Output restore waits on the mic-stop barrier in a
        second job so its ramp can overlap recognizer cleanup or batch STT.
        Always await through ``asyncio.shield``: cancelling an awaiter must
        neither cancel the underlying work nor open the door to a second
        cleanup.
        """
        if self._audio_shutdown_task is None:
            mic_stopped = threading.Event()
            self._audio_shutdown_task = asyncio.ensure_future(
                self._loop.run_in_executor(
                    None,
                    lambda: self._audio_shutdown_sync(
                        streaming,
                        cancel,
                        mic_stopped=mic_stopped,
                    ),
                )
            )
            if restore_output and duck_token is not None:
                self._output_restore_task = asyncio.ensure_future(
                    self._loop.run_in_executor(
                        None,
                        lambda: self._restore_output_after_mic_stop_sync(
                            mic_stopped,
                            duck_token,
                        ),
                    )
                )
        return self._audio_shutdown_task

    def _audio_shutdown_sync(
        self,
        streaming: bool,
        cancel: bool,
        *,
        mic_stopped: threading.Event,
    ) -> tuple[bytes | None, str | None]:
        """Executor body: stop the recorder FIRST (mic off, taps
        quiesced), then finalize or cancel streaming — even when
        recorder.stop() raises (try/finally).

        Returns ``(wav_data, final_text)``.  Only a normal release with a
        non-empty wav finalizes streaming for a result; empty wav,
        cancel/restart/history and error paths all cancel it.
        """
        app = self._app
        wav = None
        text = None
        try:
            wav = app._recorder.stop()
        finally:
            if streaming:
                try:
                    app._recorder.clear_on_audio_chunk()
                except Exception:
                    logger.exception("Failed to detach streaming audio callback")
            # Hard ordering barrier: output restoration may run concurrently
            # with recognizer finalization, but never before the mic is off.
            mic_stopped.set()
            if streaming:
                text = self._finalize_streaming_sync(cancel=cancel or not wav)
        return wav, text

    def _restore_output_after_mic_stop_sync(
        self,
        mic_stopped: threading.Event,
        duck_token: object,
    ) -> bool:
        """Restore output once recorder.stop() has crossed its finalizer."""
        mic_stopped.wait()
        return self._end_output_duck_sync(duck_token)

    def _end_output_duck_sync(self, duck_token: object) -> bool:
        """Best-effort restore that honors an explicit failure result."""
        for attempt in range(1, self._OUTPUT_RESTORE_ATTEMPTS + 1):
            try:
                result = self._app._system_output_ducker.end(duck_token)
            except Exception:
                logger.exception("Failed to restore system output volume")
                return False
            if result is not False:
                return True
            if attempt < self._OUTPUT_RESTORE_ATTEMPTS:
                time.sleep(self._OUTPUT_RESTORE_RETRY_DELAY)

        logger.error(
            "System output volume restore returned False after %d attempts",
            self._OUTPUT_RESTORE_ATTEMPTS,
        )
        return False

    async def _settle_output_restore(self) -> bool:
        """Wait for this session's restore task without cancelling it.

        Returns whether audio shutdown transferred ownership of the duck
        token to a restore task.  A false result tells the session finalizer
        to use the direct best-effort fallback instead.
        """
        task = self._output_restore_task
        if task is None:
            return False
        try:
            await asyncio.shield(task)
        except Exception:
            logger.exception("System output restore task failed")
        return True

    def _finalize_streaming_sync(self, cancel: bool) -> str | None:
        """Stop or cancel streaming, with mutual fallback.

        Streaming counts as cleaned only when one of the two paths
        succeeded; when both fail the primary error propagates so the
        caller knows the recognizer is in an unknown state.
        """
        app = self._app
        try:
            if cancel:
                app._transcriber.cancel_streaming()
                return None
            return app._transcriber.stop_streaming()
        except Exception as primary_exc:
            logger.exception(
                "Primary streaming cleanup (%s) failed",
                "cancel" if cancel else "stop",
            )
            try:
                if cancel:
                    app._transcriber.stop_streaming()
                else:
                    app._transcriber.cancel_streaming()
            except Exception:
                logger.exception("Fallback streaming cleanup also failed")
                raise primary_exc
            return None

    def _launch_recorder_start(self, *, armed: bool) -> asyncio.Future:
        """Launch recorder.start() on the executor and track the future."""
        app = self._app
        self._pending_start = asyncio.ensure_future(
            self._loop.run_in_executor(
                None, lambda: app._recorder.start(armed=armed)
            )
        )
        return self._pending_start

    async def _settle_pending_start(
        self,
        duck_token: object | None = None,
        *,
        restore_output: bool = True,
    ) -> None:
        """Await an in-flight recorder.start() and close the mic it opened.

        Cancelling the future cannot stop the executor thread, so the
        start is either awaited to completion (then shut down through the
        session's single-flight path) or abandoned via the same taint
        mechanism the start-timeout path uses.  No-op when no start is
        pending.
        """
        fut, self._pending_start = self._pending_start, None
        if fut is None:
            return
        try:
            await asyncio.wait_for(
                asyncio.shield(fut), timeout=self._START_TIMEOUT
            )
        except TimeoutError:
            self._app._recorder.mark_tainted()
            if restore_output:
                await self._restore_output_duck(duck_token)
            return
        except Exception:
            # start() raised before touching the hardware (e.g. a
            # concurrent start in flight); nothing to close.
            if restore_output:
                await self._restore_output_duck(duck_token)
            return
        if self._app._recorder.is_recording:
            # streaming=False is accurate here: streaming only attaches
            # after arm(), which never ran for a pending start.
            await asyncio.shield(
                self._ensure_audio_shutdown(
                    False,
                    cancel=True,
                    duck_token=duck_token,
                    restore_output=restore_output,
                )
            )
        elif restore_output:
            await self._restore_output_duck(duck_token)

    async def _cleanup_session_audio(
        self, streaming: bool, duck_token: object | None = None,
    ) -> None:
        """Best-effort abort cleanup via the single-flight shutdown task.

        Awaiting the shared task makes this idempotent: when a shutdown
        already ran (or is in flight), no second recorder stop and no
        second streaming cleanup can ever start.
        """
        app = self._app
        if (
            self._audio_shutdown_task is None
            and not app._recorder.is_recording
            and not streaming
        ):
            return
        try:
            await asyncio.shield(
                self._ensure_audio_shutdown(
                    streaming,
                    cancel=True,
                    duck_token=duck_token,
                )
            )
        except Exception:
            logger.exception("Session audio cleanup failed")

    # ------------------------------------------------------------------
    # Level polling
    # ------------------------------------------------------------------

    async def _poll_level(self) -> None:
        """Poll audio level and update the indicator."""
        from PyObjCTools import AppHelper

        app = self._app
        try:
            while True:
                level = app._recorder.current_level
                AppHelper.callAfter(
                    app._recording_indicator.update_level, level
                )
                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    def _show_live_overlay(self, active: bool = True) -> None:
        """Show the live transcription overlay (must be called on main thread)."""
        try:
            app = self._app
            if hasattr(app, "_live_overlay") and app._live_overlay is not None:
                self._live_overlay = app._live_overlay
            else:
                from wenzi.ui.live_transcription_overlay import (
                    LiveTranscriptionOverlay,
                )
                self._live_overlay = LiveTranscriptionOverlay()
            self._live_overlay.show(active=active)
        except Exception:
            logger.exception("Failed to show live overlay")

    def _update_live_overlay(self, text: str) -> None:
        if self._live_overlay is not None:
            self._live_overlay.update_text(text)

    def _hide_live_overlay(self) -> None:
        from PyObjCTools import AppHelper

        def _close():
            from wenzi.ui.live_transcription_overlay import LiveTranscriptionOverlay
            LiveTranscriptionOverlay.close_all()
            self._live_overlay = None
        AppHelper.callAfter(_close)

    def _reset_to_idle(self) -> None:
        """Common cleanup: hide overlays/indicator and restore idle status."""
        self._target_app = None
        self._hide_live_overlay()
        self._cancel_level_task()
        self._app._recording_indicator.hide()
        self._app._set_status("statusbar.status.ready")
        self._restore_mode()

    def _cancel_subtasks(self) -> None:
        """Cancel level polling task."""
        self._cancel_level_task()

    def _drain_actions(self) -> None:
        """Discard all pending actions from the queue."""
        while not self._actions.empty():
            try:
                self._actions.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _cancel_level_task(self) -> None:
        if self._level_task and not self._level_task.done():
            self._level_task.cancel()
            self._level_task = None

    # ------------------------------------------------------------------
    # Mode management (carried over from RecordingController)
    # ------------------------------------------------------------------

    def _apply_prefer_mode(self, mode: str) -> None:
        app = self._app
        self._saved_mode = (
            app._enhance_mode,
            app._enhancer.mode if app._enhancer else None,
            app._enhancer._enabled if app._enhancer else None,
        )
        self._switch_active_mode(mode)
        logger.info(
            "Prefer mode applied: %s, saved: %s", mode, self._saved_mode[0]
        )

    def _restore_mode(self) -> None:
        self._no_preview_override = False
        if self._saved_mode is None:
            return
        from PyObjCTools import AppHelper

        app = self._app
        orig_mode, orig_enhancer_mode, orig_enhancer_enabled = self._saved_mode
        self._saved_mode = None

        app._enhance_mode = orig_mode
        app._enhance_controller.enhance_mode = orig_mode

        if app._enhancer:
            if orig_enhancer_mode is not None:
                app._enhancer.mode = orig_enhancer_mode
            if orig_enhancer_enabled is not None:
                app._enhancer._enabled = orig_enhancer_enabled

        for m, item in app._enhance_menu_items.items():
            AppHelper.callAfter(
                lambda i=item, s=(1 if m == orig_mode else 0): i.setState_(s)
            )
        logger.info("Mode restored to: %s", orig_mode)

    def _switch_active_mode(self, mode: str) -> None:
        from wenzi.enhance.enhancer import MODE_OFF

        app = self._app
        app._enhance_mode = mode
        app._enhance_controller.enhance_mode = mode
        if app._enhancer:
            if mode == MODE_OFF:
                app._enhancer._enabled = False
            else:
                app._enhancer._enabled = True
                app._enhancer.mode = mode

    def _build_mode_list(self) -> list[tuple[str, str]]:
        from wenzi.enhance.enhancer import MODE_OFF

        app = self._app
        modes: list[tuple[str, str]] = [(MODE_OFF, "Off")]
        if app._enhancer:
            modes.extend(app._enhancer.available_modes)
        return modes

    def _navigate_mode(self, delta: int) -> None:
        if self._no_preview_override:
            return
        from PyObjCTools import AppHelper

        modes = self._build_mode_list()
        if len(modes) <= 1:
            return
        current = self._app._enhance_mode
        idx = next(
            (i for i, (mid, _) in enumerate(modes) if mid == current), -1
        )
        new_idx = idx + delta
        if idx < 0 or new_idx < 0 or new_idx >= len(modes):
            return

        new_mode = modes[new_idx][0]
        if self._saved_mode is None:
            self._apply_prefer_mode(new_mode)
        else:
            self._switch_active_mode(new_mode)

        label = modes[new_idx][1]
        AppHelper.callAfter(
            self._app._recording_indicator.update_mode,
            label,
        )
        logger.info(
            "Mode nav %s → %s", "prev" if delta < 0 else "next", new_mode
        )

    def _get_mode_label(self) -> str | None:
        """Return the current enhance-mode label, or None if not applicable."""
        if self._no_preview_override:
            return None
        modes = self._build_mode_list()
        if len(modes) <= 1:
            return None
        current = self._app._enhance_mode
        idx = next(
            (i for i, (mid, _) in enumerate(modes) if mid == current), -1
        )
        if idx < 0:
            return None
        return modes[idx][1]

    def _show_mode_on_indicator(self) -> None:
        label = self._get_mode_label()
        if label is None:
            return
        from PyObjCTools import AppHelper

        AppHelper.callAfter(
            self._app._recording_indicator.update_mode,
            label,
        )

    # ------------------------------------------------------------------
    # Feedback toggles (kept for menu item callbacks)
    # ------------------------------------------------------------------

    def on_sound_feedback_toggle(self, sender) -> None:
        app = self._app
        app._sound_manager.enabled = not app._sound_manager.enabled
        sender.state = 1 if app._sound_manager.enabled else 0
        fb_cfg = app._config.setdefault("feedback", {})
        fb_cfg["sound_enabled"] = app._sound_manager.enabled
        save_config(app._config, app._config_path)

    def on_visual_indicator_toggle(self, sender) -> None:
        app = self._app
        app._recording_indicator.enabled = not app._recording_indicator.enabled
        sender.state = 1 if app._recording_indicator.enabled else 0
        fb_cfg = app._config.setdefault("feedback", {})
        fb_cfg["visual_indicator"] = app._recording_indicator.enabled
        save_config(app._config, app._config_path)

    # ------------------------------------------------------------------
    # Scripting events
    # ------------------------------------------------------------------

    def _fire_scripting_event(self, event_name: str, **kwargs) -> None:
        fire_scripting_event(self._app, event_name, **kwargs)

    # ------------------------------------------------------------------
    # Voice input initialization (delegated to main thread)
    # ------------------------------------------------------------------

    def _try_enable_voice_input(self) -> None:
        """Attempt to initialize voice input (runs on main thread via callAfter)."""
        app = self._app
        try:
            from wenzi.transcription.apple import check_siri_available

            ok, _ = check_siri_available()
            if not ok:
                from wenzi.transcription.apple import prompt_siri_setup
                choice = prompt_siri_setup()
                app._handle_dictation_setup_choice(choice)
                return

            app._transcriber.initialize()
            app._voice_input_available = True
            app._set_status("statusbar.status.ready")
            logger.info("Voice input enabled after deferred initialization")
        except Exception:
            logger.debug("Deferred voice init failed, prompting user")
            from wenzi.transcription.apple import prompt_siri_setup
            choice = prompt_siri_setup()
            app._handle_dictation_setup_choice(choice)
