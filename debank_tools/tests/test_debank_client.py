"""Tests for the DeBank CLI client."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLIENT_PATH = ROOT / "cli" / "debank_client.py"
spec = importlib.util.spec_from_file_location("debank_client", CLIENT_PATH)
debank_client = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["debank_client"] = debank_client
spec.loader.exec_module(debank_client)


def test_normalize_address_lowercases():
    assert debank_client.normalize_address("0xABCDEFabcdefABCDEFabcdefABCDEFabcdefABCD") == "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"


def test_normalize_address_rejects_invalid():
    with pytest.raises(debank_client.DebankError):
        debank_client.normalize_address("abc")


class FakeClient:
    def total_balance(self, address):
        return {"total_usd_value": 1234.5, "address": address}

    def used_chains(self, address):
        return [{"id": "eth", "name": "Ethereum"}]

    def tokens(self, address):
        return [
            {"chain": "eth", "symbol": "ETH", "name": "Ether", "amount": 2, "price": 3000},
            {"chain": "eth", "symbol": "USDC", "name": "USD Coin", "amount": 100, "price": 1},
        ]

    def protocols(self, address):
        return [
            {"chain": "eth", "id": "aave", "name": "Aave", "net_usd_value": 500},
            {"chain": "eth", "id": "compound", "name": "Compound", "asset_usd_value": 100},
        ]


def test_build_summary_sorts_top_assets():
    summary = debank_client.build_summary(
        FakeClient(),
        "0xABCDEFabcdefABCDEFabcdefABCDEFabcdefABCD",
        top=1,
    )

    assert summary["address"] == "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    assert summary["top_tokens"] == [
        {
            "chain": "eth",
            "symbol": "ETH",
            "name": "Ether",
            "amount": 2.0,
            "price": 3000.0,
            "usd_value": 6000.0,
        }
    ]
    assert summary["top_protocols"][0]["name"] == "Aave"


def test_main_requires_key(capsys):
    code = debank_client.main(["units"])
    captured = capsys.readouterr()

    assert code == 1
    assert "Missing access key" in captured.err


def test_main_units_uses_env_key(monkeypatch, capsys):
    class FakeDebankClient:
        def __init__(self, config):
            self.config = config

        def units(self):
            return {"used": 1, "remaining": 99}

    monkeypatch.setenv("DEBANK_ACCESS_KEY", "test-key")
    monkeypatch.setattr(debank_client, "DebankClient", FakeDebankClient)

    code = debank_client.main(["units"])
    captured = capsys.readouterr()

    assert code == 0
    assert '"remaining": 99' in captured.out
