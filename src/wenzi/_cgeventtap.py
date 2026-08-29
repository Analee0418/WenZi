"""Low-level ctypes bindings for CGEventTap — no PyObjC bridge.

Using ctypes instead of PyObjC for CGEventTapCreate avoids the PyObjC
callback bridge retaining CGEventRef wrappers indefinitely.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import logging as _logging
import threading as _threading
import time as _time
from ctypes import CFUNCTYPE, c_bool, c_int32, c_int64, c_uint32, c_uint64, c_void_p

# ---------------------------------------------------------------------------
# Load frameworks
# ---------------------------------------------------------------------------
_cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreGraphics"))
_cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))

# ---------------------------------------------------------------------------
# Callback type
# ---------------------------------------------------------------------------
CGEventTapCallBack = CFUNCTYPE(c_void_p, c_void_p, c_uint32, c_void_p, c_void_p)

# ---------------------------------------------------------------------------
# Constants (hardcoded — no Quartz import)
# ---------------------------------------------------------------------------
kCGSessionEventTap = 1
kCGHeadInsertEventTap = 0

kCGEventTapOptionDefault = 0
kCGEventTapOptionListenOnly = 1

kCGEventKeyDown = 10
kCGEventKeyUp = 11
kCGEventFlagsChanged = 12

kCGEventTapDisabledByTimeout = 0xFFFFFFFE

kCGKeyboardEventKeycode = 9

kCGAnnotatedSessionEventTap = 2

kCGEventSourceStateCombinedSessionState = 0

kCGEventFlagMaskCommand = 1 << 20
kCGEventFlagMaskControl = 1 << 18
kCGEventFlagMaskAlternate = 1 << 19
kCGEventFlagMaskShift = 1 << 17

kCFRunLoopDefaultMode = c_void_p.in_dll(_cf, "kCFRunLoopDefaultMode")

# ---------------------------------------------------------------------------
# Function signatures
# ---------------------------------------------------------------------------

# CGEventTapCreate(tap, place, options, eventsOfInterest, callback, userInfo)
_cg.CGEventTapCreate.restype = c_void_p
_cg.CGEventTapCreate.argtypes = [
    c_uint32,           # CGEventTapLocation
    c_uint32,           # CGEventTapPlacement
    c_uint32,           # CGEventTapOptions
    c_uint64,           # CGEventMask
    CGEventTapCallBack, # callback
    c_void_p,           # userInfo
]

# CGEventTapEnable(tap, enable)
_cg.CGEventTapEnable.restype = None
_cg.CGEventTapEnable.argtypes = [c_void_p, c_bool]

# CGEventGetIntegerValueField(event, field) -> int64
_cg.CGEventGetIntegerValueField.restype = c_int64
_cg.CGEventGetIntegerValueField.argtypes = [c_void_p, c_uint32]

# CGEventGetFlags(event) -> uint64
_cg.CGEventGetFlags.restype = c_uint64
_cg.CGEventGetFlags.argtypes = [c_void_p]

# CGEventSetFlags(event, flags)
_cg.CGEventSetFlags.restype = None
_cg.CGEventSetFlags.argtypes = [c_void_p, c_uint64]

# CGEventSourceFlagsState(stateID) -> uint64
_cg.CGEventSourceFlagsState.restype = c_uint64
_cg.CGEventSourceFlagsState.argtypes = [c_int32]

# CGEventCreateKeyboardEvent(source, virtualKey, keyDown) -> CGEventRef
_cg.CGEventCreateKeyboardEvent.restype = c_void_p
_cg.CGEventCreateKeyboardEvent.argtypes = [c_void_p, c_uint32, c_bool]

# CGEventPost(tap, event)
_cg.CGEventPost.restype = None
_cg.CGEventPost.argtypes = [c_uint32, c_void_p]

# CFMachPortCreateRunLoopSource(allocator, port, order) -> CFRunLoopSourceRef
_cf.CFMachPortCreateRunLoopSource.restype = c_void_p
_cf.CFMachPortCreateRunLoopSource.argtypes = [c_void_p, c_void_p, c_int64]

# CFRunLoopGetCurrent() -> CFRunLoopRef
_cf.CFRunLoopGetCurrent.restype = c_void_p
_cf.CFRunLoopGetCurrent.argtypes = []

# CFRunLoopAddSource(rl, source, mode)
_cf.CFRunLoopAddSource.restype = None
_cf.CFRunLoopAddSource.argtypes = [c_void_p, c_void_p, c_void_p]

# CFRunLoopRun()
_cf.CFRunLoopRun.restype = None
_cf.CFRunLoopRun.argtypes = []

# CFRunLoopStop(rl)
_cf.CFRunLoopStop.restype = None
_cf.CFRunLoopStop.argtypes = [c_void_p]

# CFRelease(cf)
_cf.CFRelease.restype = None
_cf.CFRelease.argtypes = [c_void_p]

# ---------------------------------------------------------------------------
# Module-level Python functions
# ---------------------------------------------------------------------------


def CGEventTapCreate(tap, place, options, events_of_interest, callback, user_info):
    return _cg.CGEventTapCreate(tap, place, options, events_of_interest, callback, user_info)


def CGEventTapEnable(tap, enable):
    _cg.CGEventTapEnable(tap, enable)


def CGEventGetIntegerValueField(event, field):
    return _cg.CGEventGetIntegerValueField(event, field)


def CGEventGetFlags(event):
    return _cg.CGEventGetFlags(event)


def CGEventSetFlags(event, flags):
    _cg.CGEventSetFlags(event, flags)


def CGEventSourceFlagsState(state_id):
    return _cg.CGEventSourceFlagsState(state_id)


def CGEventCreateKeyboardEvent(source, virtual_key, key_down):
    return _cg.CGEventCreateKeyboardEvent(source, virtual_key, key_down)


def CGEventPost(tap, event):
    _cg.CGEventPost(tap, event)


def CFMachPortCreateRunLoopSource(allocator, port, order):
    return _cf.CFMachPortCreateRunLoopSource(allocator, port, order)


def CFRunLoopGetCurrent():
    return _cf.CFRunLoopGetCurrent()


def CFRunLoopAddSource(rl, source, mode):
    _cf.CFRunLoopAddSource(rl, source, mode)


def CFRunLoopRun():
    _cf.CFRunLoopRun()


def CFRunLoopStop(rl):
    _cf.CFRunLoopStop(rl)


def CFRelease(cf):
    _cf.CFRelease(cf)


def CGEventMaskBit(event_type):
    """Pure Python implementation of CGEventMaskBit."""
    return 1 << event_type


# ---------------------------------------------------------------------------
# CGEventTapRunner — shared lifecycle helper
# ---------------------------------------------------------------------------

_runner_logger = _logging.getLogger(__name__)


class CGEventTapRunner:
    """Manages a CGEventTap on a background thread with proper cleanup.

    Handles the boilerplate: create tap, create run-loop source, run the
    CFRunLoop on a daemon thread, and tear everything down with CFRelease
    on stop().  Consumers only supply a callback and event mask.
    """


    _STATE_IDLE = "idle"
    _STATE_STARTING = "starting"
    _STATE_RUNNING = "running"
    _STATE_STOPPING = "stopping"
    _STATE_REAPING = "reaping"

    # How long stop() keeps re-issuing CFRunLoopStop + join before giving
    # up and letting the thread's own reaper finish the cleanup later.
    _STOP_JOIN_TIMEOUT = 2.0

    def __init__(self) -> None:
        self.tap = None
        self._source = None
        self._loop = None
        self._thread: _threading.Thread | None = None
        self._ctypes_cb = None
        self._ready = _threading.Event()
        # Explicit lifecycle: IDLE -> STARTING -> RUNNING -> STOPPING ->
        # IDLE.  Guarded by _lifecycle; join/wait never happen inside it.
        self._lifecycle = _threading.Lock()
        self._state = self._STATE_IDLE
        # Generation identity: the run thread's reaper only cleans up its
        # own generation and can never touch a newer instance's state.
        self._gen = 0

    @property
    def running(self) -> bool:
        return self._state in (self._STATE_STARTING, self._STATE_RUNNING)

    def start(
        self,
        mask: int,
        callback,
        *,
        option: int = kCGEventTapOptionDefault,
        on_create_failed=None,
    ) -> None:
        """Start the tap on a background thread.

        *callback* receives ``(proxy, event_type, event, refcon)`` and must
        return the event (pass-through) or ``None`` (swallow / listen-only).
        *on_create_failed* is called (on the bg thread) if
        ``CGEventTapCreate`` returns NULL.

        Raises RuntimeError unless the runner is IDLE: a previous
        instance that has not finished cleaning up still owns native
        objects, and replacing its ctypes callback would be a
        use-after-free under the live run loop.
        """
        with self._lifecycle:
            if self._state != self._STATE_IDLE:
                raise RuntimeError(
                    "CGEventTapRunner.start() while "
                    f"{self._state}; previous instance not cleaned up yet"
                )
            self._state = self._STATE_STARTING
            self._gen += 1
            gen = self._gen
            ready = _threading.Event()
            self._ready = ready

            def _raw_cb(proxy, event_type, event, refcon):
                return callback(proxy, event_type, event, refcon) or 0

            # The closure keeps this generation's ctypes callback alive
            # for as long as ITS thread may use it, independent of any
            # newer instance overwriting self._ctypes_cb.
            cb_ref = CGEventTapCallBack(_raw_cb)
            self._ctypes_cb = cb_ref

        def _run():
            tap = None
            source = None
            entered_loop = False
            try:
                tap = CGEventTapCreate(
                    kCGSessionEventTap, kCGHeadInsertEventTap,
                    option, mask, cb_ref, None,
                )
                if not tap:
                    _runner_logger.warning(
                        "CGEventTapCreate failed — check Accessibility "
                        "permissions in System Settings > Privacy & "
                        "Security > Accessibility"
                    )
                    if on_create_failed is not None:
                        on_create_failed()
                    return
                source = CFMachPortCreateRunLoopSource(None, tap, 0)
                loop = CFRunLoopGetCurrent()
                CFRunLoopAddSource(loop, source, kCFRunLoopDefaultMode.value)
                CGEventTapEnable(tap, True)
                with self._lifecycle:
                    if (
                        self._gen == gen
                        and self._state == self._STATE_STARTING
                    ):
                        self.tap = tap
                        self._source = source
                        self._loop = loop
                        self._state = self._STATE_RUNNING
                        entered_loop = True
                    # else: stop() arrived during setup — skip the loop,
                    # fall through to the reaper below.
                _runner_logger.info(
                    "CGEventTap created (tap=%#x, option=%d, mask=0x%x)",
                    tap, option, mask,
                )
                ready.set()
                if entered_loop:
                    CFRunLoopRun()
            except Exception:
                _runner_logger.exception("CGEventTap setup failed")
            finally:
                ready.set()
                # Reaper, three phases:
                #   1 (locked)   claim REAPING for our generation and
                #                atomically detach the shared handles — a
                #                concurrent stop() can never again reach a
                #                native object that is about to be freed;
                #   2 (unlocked) disable/release the LOCAL handles;
                #   3 (locked)   clear thread/callback and go IDLE.
                with self._lifecycle:
                    if self._gen == gen:
                        self._state = self._STATE_REAPING
                        self.tap = None
                        self._source = None
                        self._loop = None
                try:
                    if tap:
                        try:
                            CGEventTapEnable(tap, False)
                        except Exception:
                            _runner_logger.debug(
                                "Tap disable failed", exc_info=True
                            )
                    if source:
                        try:
                            CFRelease(source)
                        except Exception:
                            _runner_logger.debug(
                                "CFRelease(source) failed", exc_info=True
                            )
                    if tap:
                        try:
                            CFRelease(tap)
                        except Exception:
                            _runner_logger.debug(
                                "CFRelease(tap) failed", exc_info=True
                            )
                finally:
                    with self._lifecycle:
                        # Only our own generation: a newer instance's
                        # state must never be cleared by an old reaper.
                        if self._gen == gen:
                            self._thread = None
                            self._ctypes_cb = None
                            self._state = self._STATE_IDLE

        thread = _threading.Thread(
            target=_run, name=f"cgeventtap-runloop-{gen}", daemon=True,
        )
        with self._lifecycle:
            self._thread = thread
        thread.start()

    def wait_ready(self, timeout: float = 2.0) -> None:
        """Block until the background thread has created the tap (or failed)."""
        self._ready.wait(timeout)

    def stop(self) -> None:
        """Request shutdown; never frees native objects itself.

        Ownership of every CFRelease lives in the run thread's reaper, so
        the release happens exactly once and provably after CFRunLoopRun
        returned.  stop() only flips the state, disables the tap, kicks
        the loop, and (outside the lifecycle lock) waits briefly for the
        thread; a thread that misses the deadline reclaims itself later.
        Safe to call concurrently and from the tap callback itself.
        """
        with self._lifecycle:
            if self._state == self._STATE_IDLE:
                return
            if self._state in (self._STATE_STARTING, self._STATE_RUNNING):
                self._state = self._STATE_STOPPING
            # In REAPING the shared handles are already detached (None) —
            # nothing below can touch a freed native object.
            gen = self._gen
            thread = self._thread
            tap = self.tap
            loop = self._loop
            ready = self._ready

        if thread is not None and thread is _threading.current_thread():
            # Called from the tap callback on our own run-loop thread:
            # disable the tap and ask the loop to exit after this
            # callback returns — our own finally reaps everything.
            try:
                if tap:
                    CGEventTapEnable(tap, False)
                if loop:
                    CFRunLoopStop(loop)
            except Exception:
                _runner_logger.debug("Self-stop error", exc_info=True)
            return

        # Let setup finish so tap/loop below hold their final values.
        ready.wait(2.0)
        # The generation check, the handle read AND the native disable
        # form ONE linearized interval under the lifecycle lock: the
        # reaper's phase 1 takes the same lock, so it cannot detach and
        # free the tap while we are inside CGEventTapEnable — and once it
        # has reaped, we read None and touch nothing.
        with self._lifecycle:
            if self._gen == gen:
                tap_now = self.tap
                if tap_now:
                    try:
                        CGEventTapEnable(tap_now, False)
                    except Exception:
                        _runner_logger.debug(
                            "CGEventTapRunner: error during disable",
                            exc_info=True,
                        )
                thread = self._thread or thread

        if thread is None:
            return
        # The thread may be anywhere between ready.set() and
        # CFRunLoopRun(); re-issue CFRunLoopStop until it exits.  This
        # join deliberately happens OUTSIDE the lifecycle lock.
        deadline = _time.monotonic() + self._STOP_JOIN_TIMEOUT
        while thread.is_alive() and _time.monotonic() < deadline:
            # Re-read the loop handle under the lock every iteration: the
            # reaper detaches it (phase 1) before freeing, so we can never
            # kick a loop that is being (or has been) released.
            with self._lifecycle:
                loop_now = self._loop if self._gen == gen else None
            try:
                if loop_now:
                    CFRunLoopStop(loop_now)
            except Exception:
                _runner_logger.debug("CFRunLoopStop failed", exc_info=True)
            thread.join(timeout=0.1)
        if thread.is_alive():
            _runner_logger.error(
                "CGEventTap run-loop thread did not exit in time; it will "
                "reap its own native objects when it exits"
            )
