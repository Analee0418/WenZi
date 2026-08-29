"""Tests for the app-wide exclusive-operation guard."""

from __future__ import annotations

from wenzi.op_guard import OpGuard


class TestOpGuard:
    def test_begin_and_end(self):
        g = OpGuard()
        assert g.busy is False
        token = g.try_begin("recording")
        assert token is not None
        assert g.busy is True
        assert g.owner_name == "recording"
        g.end(token)
        assert g.busy is False
        assert g.owner_name is None

    def test_second_claim_refused(self):
        g = OpGuard()
        t1 = g.try_begin("a")
        assert g.try_begin("b") is None
        g.end(t1)
        assert g.try_begin("b") is not None

    def test_stale_release_cannot_free_newer_claim(self):
        """A duplicate or late end() must never release a claim it does
        not own — even one started under the same name."""
        g = OpGuard()
        t1 = g.try_begin("model-switch")
        g.end(t1)
        t2 = g.try_begin("model-switch")

        g.end(t1)  # stale duplicate release from the previous operation
        assert g.busy is True  # t2 still holds the slot

        g.end(t2)
        assert g.busy is False

    def test_end_none_is_noop(self):
        g = OpGuard()
        token = g.try_begin("x")
        g.end(None)
        assert g.busy is True
        g.end(token)
        assert g.busy is False
