#!/usr/bin/env python3
"""Small DeBank OpenAPI CLI for wallet analysis.

Authentication:
    export DEBANK_ACCESS_KEY=...
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

BASE_URL = "https://pro-openapi.debank.com"
ACCESS_KEY_ENV = "DEBANK_ACCESS_KEY"
ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
DEFAULT_TIMEOUT = 20


class DebankError(RuntimeError):
    """Raised for DeBank request or response failures."""


@dataclass(frozen=True)
class ClientConfig:
    access_key: str
    base_url: str = BASE_URL
    timeout: int = DEFAULT_TIMEOUT


class DebankClient:
    """Minimal DeBank OpenAPI client."""

    def __init__(self, config: ClientConfig) -> None:
        self.config = config

    def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = urllib.parse.urlencode({k: v for k, v in (params or {}).items() if v is not None})
        url = self.config.base_url.rstrip("/") + path
        if query:
            url += "?" + query
        request = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "AccessKey": self.config.access_key,
                "User-Agent": "debank-tools/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
                body = response.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise DebankError(f"DeBank HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise DebankError(f"DeBank request failed: {exc.reason}") from exc

        try:
            return json.loads(body)
        except json.JSONDecodeError as exc:
            raise DebankError("DeBank returned non-JSON response") from exc

    def units(self) -> Any:
        return self.get("/v1/account/units")

    def total_balance(self, address: str) -> Any:
        return self.get("/v1/user/total_balance", {"id": normalize_address(address)})

    def used_chains(self, address: str) -> Any:
        return self.get("/v1/user/used_chain_list", {"id": normalize_address(address)})

    def tokens(self, address: str, *, is_all: bool = False) -> Any:
        return self.get("/v1/user/all_token_list", {"id": normalize_address(address), "is_all": str(is_all).lower()})

    def protocols(self, address: str, *, complex_positions: bool = False) -> Any:
        endpoint = "/v1/user/all_complex_protocol_list" if complex_positions else "/v1/user/all_simple_protocol_list"
        return self.get(endpoint, {"id": normalize_address(address)})

    def approvals(self, address: str, *, chain_id: str | None = None) -> Any:
        params = {"id": normalize_address(address)}
        if chain_id:
            params["chain_id"] = chain_id
        return self.get("/v1/user/token_authorized_list", params)

    def history(self, address: str, *, chain_id: str, page_count: int = 20, start_time: int | None = None) -> Any:
        return self.get(
            "/v1/user/history_list",
            {
                "id": normalize_address(address),
                "chain_id": chain_id,
                "page_count": page_count,
                "start_time": start_time,
            },
        )


def normalize_address(address: str) -> str:
    value = (address or "").strip()
    if not ADDRESS_RE.match(value):
        raise DebankError(f"Invalid EVM address: {address!r}")
    return value.lower()


def build_summary(client: DebankClient, address: str, *, top: int = 10) -> dict[str, Any]:
    """Fetch and summarize common wallet portfolio data."""
    normalized = normalize_address(address)
    total = client.total_balance(normalized)
    chains = client.used_chains(normalized)
    tokens = client.tokens(normalized)
    protocols = client.protocols(normalized)

    top_tokens = sorted(
        [t for t in tokens if isinstance(t, dict)],
        key=lambda item: float(item.get("amount") or 0) * float(item.get("price") or 0),
        reverse=True,
    )[:top]
    top_protocols = sorted(
        [p for p in protocols if isinstance(p, dict)],
        key=lambda item: float(item.get("net_usd_value") or item.get("asset_usd_value") or 0),
        reverse=True,
    )[:top]

    return {
        "address": normalized,
        "total_balance": total,
        "used_chains": chains,
        "top_tokens": [_token_summary(t) for t in top_tokens],
        "top_protocols": [_protocol_summary(p) for p in top_protocols],
    }


def _token_summary(token: dict[str, Any]) -> dict[str, Any]:
    amount = float(token.get("amount") or 0)
    price = float(token.get("price") or 0)
    return {
        "chain": token.get("chain"),
        "symbol": token.get("symbol"),
        "name": token.get("name"),
        "amount": amount,
        "price": price,
        "usd_value": amount * price,
    }


def _protocol_summary(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "chain": protocol.get("chain"),
        "id": protocol.get("id"),
        "name": protocol.get("name"),
        "site_url": protocol.get("site_url"),
        "net_usd_value": protocol.get("net_usd_value") or protocol.get("asset_usd_value") or 0,
    }


def make_client(args: argparse.Namespace) -> DebankClient:
    access_key = args.access_key or os.environ.get(ACCESS_KEY_ENV, "")
    if not access_key:
        raise DebankError(f"Missing access key. Set {ACCESS_KEY_ENV} or pass --access-key.")
    return DebankClient(ClientConfig(access_key=access_key, base_url=args.base_url, timeout=args.timeout))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Query DeBank OpenAPI wallet data")
    parser.add_argument("--access-key", default="", help=f"DeBank AccessKey. Defaults to ${ACCESS_KEY_ENV}.")
    parser.add_argument("--base-url", default=BASE_URL, help="DeBank OpenAPI base URL")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="HTTP timeout in seconds")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("units", help="Show API units usage")

    summary = sub.add_parser("summary", help="Show wallet portfolio summary")
    summary.add_argument("address")
    summary.add_argument("--top", type=int, default=10)

    tokens = sub.add_parser("tokens", help="Show wallet token list")
    tokens.add_argument("address")
    tokens.add_argument("--all", action="store_true", help="Ask DeBank to include hidden/small tokens")

    protocols = sub.add_parser("protocols", help="Show wallet protocols")
    protocols.add_argument("address")
    protocols.add_argument("--complex", action="store_true", help="Use complex protocol positions endpoint")

    approvals = sub.add_parser("approvals", help="Show token approvals")
    approvals.add_argument("address")
    approvals.add_argument("--chain", default="", help="Optional chain id, e.g. eth")

    history = sub.add_parser("history", help="Show wallet transaction history for one chain")
    history.add_argument("address")
    history.add_argument("--chain", required=True, help="Chain id, e.g. eth")
    history.add_argument("--page-count", type=int, default=20)
    history.add_argument("--start-time", type=int)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        client = make_client(args)
        if args.command == "units":
            result = client.units()
        elif args.command == "summary":
            result = build_summary(client, args.address, top=args.top)
        elif args.command == "tokens":
            result = client.tokens(args.address, is_all=args.all)
        elif args.command == "protocols":
            result = client.protocols(args.address, complex_positions=args.complex)
        elif args.command == "approvals":
            result = client.approvals(args.address, chain_id=args.chain or None)
        elif args.command == "history":
            result = client.history(args.address, chain_id=args.chain, page_count=args.page_count, start_time=args.start_time)
        else:
            parser.error(f"Unknown command: {args.command}")
            return 2
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except DebankError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
