"""Integration tests for Direct-mode streaming enhancement.

These call the REAL RecordingFlow._run_direct_single_stream /
_run_direct_chain_stream coroutines (no replicated logic) against a mock
app on the shared asyncio loop.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

import wenzi.async_loop as async_loop
from wenzi.controllers.recording_flow import RecordingFlow


@pytest.fixture(autouse=True)
def _fresh_loop():
    async_loop.shutdown_sync(timeout=2)
    yield
    async_loop.shutdown_sync(timeout=2)


@pytest.fixture
def mock_app():
    app = MagicMock()
    app._streaming_overlay = MagicMock()
    app._usage_stats = MagicMock()
    app._enhancer = MagicMock()
    app._enhancer.mode = "proofread"
    app._enhancer.get_mode_definition.return_value = None
    return app


@pytest.fixture
def flow(mock_app):
    return RecordingFlow(mock_app)


def _make_async_gen(chunks):
    async def gen():
        for item in chunks:
            yield item

    return gen()


def run(coro):
    return async_loop.submit(coro).result(timeout=10)


class TestRunDirectSingleStream:
    def test_collects_chunks_and_returns_text(self, flow, mock_app):
        mock_app._enhancer.enhance_stream.return_value = _make_async_gen([
            ("Hello ", None, False),
            ("world", None, False),
            ("", {"total_tokens": 5}, False),
        ])

        result, fell_back = run(
            flow._run_direct_single_stream("orig", asyncio.Event())
        )

        assert result == "Hello world"
        assert fell_back is False
        assert mock_app._streaming_overlay.append_text.call_count == 2

    def test_thinking_then_content(self, flow, mock_app):
        mock_app._enhancer.enhance_stream.return_value = _make_async_gen([
            ("thinking...", None, True),
            ("Result", None, False),
        ])

        result, fell_back = run(
            flow._run_direct_single_stream("orig", asyncio.Event())
        )

        assert result == "Result"
        assert fell_back is False
        mock_app._streaming_overlay.append_thinking_text.assert_called()
        # Content after thinking clears the thinking view first
        mock_app._streaming_overlay.clear_text.assert_called()

    def test_retry_marker_shows_status(self, flow, mock_app):
        mock_app._enhancer.enhance_stream.return_value = _make_async_gen([
            ("(Connection timed out, retrying in 2s — 1/3...)\n", None, "retry"),
            ("Result", None, False),
        ])

        result, fell_back = run(
            flow._run_direct_single_stream("orig", asyncio.Event())
        )

        assert result == "Result"
        assert fell_back is False
        mock_app._streaming_overlay.set_status.assert_called()

    def test_cancel_stops_consumption(self, flow, mock_app):
        cancel_event = asyncio.Event()
        consumed: list[str] = []

        async def _gen():
            consumed.append("a")
            yield "part-a", None, False
            cancel_event.set()
            # Give the canceller a chance to win the race
            await asyncio.sleep(0.2)
            consumed.append("b")
            yield "part-b", None, False

        mock_app._enhancer.enhance_stream.return_value = _gen()

        result, fell_back = run(
            flow._run_direct_single_stream("orig", cancel_event)
        )

        # Cancelled mid-stream: only the first chunk was collected
        assert result == "part-a"
        assert fell_back is False
        assert consumed == ["a"]

    def test_empty_stream_falls_back_to_original(self, flow, mock_app):
        mock_app._enhancer.enhance_stream.return_value = _make_async_gen([
            ("", {"total_tokens": 1}, False),
        ])

        result, fell_back = run(
            flow._run_direct_single_stream("original", asyncio.Event())
        )

        assert result == "original"
        assert fell_back is False

    def test_records_token_usage_and_sets_complete(self, flow, mock_app):
        usage = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
        mock_app._enhancer.enhance_stream.return_value = _make_async_gen([
            ("text", None, False),
            ("", usage, False),
        ])

        run(flow._run_direct_single_stream("orig", asyncio.Event()))

        mock_app._usage_stats.record_token_usage.assert_called_once_with(usage)
        mock_app._streaming_overlay.set_complete.assert_called_once_with(usage)

    def test_timeout_fallback_returns_original(self, flow, mock_app):
        mock_app._enhancer.enhance_stream.return_value = _make_async_gen([
            ("partial", None, False),
            ("orig", None, "timeout"),
        ])
        flow._show_error_alert = MagicMock()

        result, fell_back = run(
            flow._run_direct_single_stream("orig", asyncio.Event())
        )

        assert result == "orig"
        assert fell_back is True
        # Fallback must never be marked complete
        mock_app._streaming_overlay.set_complete.assert_not_called()


class TestRunDirectChainStream:
    def _setup_steps(self, mock_app):
        step1 = MagicMock()
        step1.label = "Proofread"
        step2 = MagicMock()
        step2.label = "Translate"
        mock_app._enhancer.get_mode_definition.side_effect = (
            lambda mode_id: {"s1": step1, "s2": step2}.get(mode_id)
        )

    def test_chain_runs_steps_sequentially(self, flow, mock_app):
        self._setup_steps(mock_app)
        inputs: list[str] = []

        def _make_stream(text, input_context=None):
            inputs.append(text)
            return _make_async_gen([(f"{text}+", None, False)])

        mock_app._enhancer.enhance_stream.side_effect = _make_stream

        result, fell_back = run(
            flow._run_direct_chain_stream("orig", ["s1", "s2"], asyncio.Event())
        )

        assert inputs == ["orig", "orig+"]
        assert result == "orig++"
        assert fell_back is False

    def test_chain_restores_original_mode(self, flow, mock_app):
        self._setup_steps(mock_app)
        mock_app._enhancer.mode = "chain-mode"
        mock_app._enhancer.enhance_stream.side_effect = (
            lambda text, input_context=None: _make_async_gen(
                [("x", None, False)]
            )
        )

        run(flow._run_direct_chain_stream("orig", ["s1"], asyncio.Event()))

        assert mock_app._enhancer.mode == "chain-mode"

    def test_chain_empty_step_keeps_previous_text(self, flow, mock_app):
        self._setup_steps(mock_app)
        streams = iter([
            [("improved", None, False)],
            [("", None, False)],  # step 2 returns nothing
        ])
        mock_app._enhancer.enhance_stream.side_effect = (
            lambda text, input_context=None: _make_async_gen(next(streams))
        )

        result, fell_back = run(
            flow._run_direct_chain_stream("orig", ["s1", "s2"], asyncio.Event())
        )

        assert result == "improved"
        assert fell_back is False

    def test_chain_cancel_stops_before_next_step(self, flow, mock_app):
        self._setup_steps(mock_app)
        cancel_event = asyncio.Event()
        inputs: list[str] = []

        def _make_stream(text, input_context=None):
            inputs.append(text)

            async def _gen():
                yield "step1-out", None, False
                cancel_event.set()

            return _gen()

        mock_app._enhancer.enhance_stream.side_effect = _make_stream

        run(flow._run_direct_chain_stream("orig", ["s1", "s2"], cancel_event))

        assert inputs == ["orig"]  # step 2 never ran

    def test_chain_accumulates_usage(self, flow, mock_app):
        self._setup_steps(mock_app)
        streams = iter([
            [("a", {"prompt_tokens": 1, "completion_tokens": 1,
                    "total_tokens": 2}, False)],
            [("b", {"prompt_tokens": 2, "completion_tokens": 2,
                    "total_tokens": 4}, False)],
        ])
        mock_app._enhancer.enhance_stream.side_effect = (
            lambda text, input_context=None: _make_async_gen(next(streams))
        )

        run(flow._run_direct_chain_stream("orig", ["s1", "s2"], asyncio.Event()))

        totals = mock_app._streaming_overlay.set_complete.call_args.args[0]
        assert totals["total_tokens"] == 6
        assert mock_app._usage_stats.record_token_usage.call_count == 2

    def test_chain_step_failure_aborts_and_falls_back(self, flow, mock_app):
        self._setup_steps(mock_app)
        inputs: list[str] = []

        def _make_stream(text, input_context=None):
            inputs.append(text)
            return _make_async_gen([
                ("partial", None, False),
                ("orig", None, "timeout"),
            ])

        mock_app._enhancer.enhance_stream.side_effect = _make_stream
        flow._show_error_alert = MagicMock()

        result, fell_back = run(
            flow._run_direct_chain_stream("orig", ["s1", "s2"], asyncio.Event())
        )

        assert fell_back is True
        assert result == "orig"
        assert inputs == ["orig"]  # aborted before step 2
        mock_app._streaming_overlay.set_complete.assert_not_called()
