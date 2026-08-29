"""App-wide exclusive-operation guard (recording / model switch / preview)."""

from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)


class OpGuard:
    """Mutual exclusion with per-claim release tokens.

    A successful try_begin() returns an opaque token; only that exact
    token can release the slot.  This makes stale releases harmless: an
    operation that ends twice, or ends late, can never free a slot that
    has since been claimed by another operation — even one that was
    started under the same name.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._token: object | None = None
        self._name: str | None = None

    @property
    def busy(self) -> bool:
        return self._token is not None

    @property
    def owner_name(self) -> str | None:
        return self._name

    def try_begin(self, name: str) -> object | None:
        """Claim the slot.  Returns a release token, or None when busy."""
        with self._lock:
            if self._token is not None:
                logger.debug(
                    "Operation %r refused: %r in progress", name, self._name
                )
                return None
            token = object()
            self._token = token
            self._name = name
            return token

    def end(self, token: object | None) -> None:
        """Release the slot if *token* is the claim that holds it."""
        if token is None:
            return
        with self._lock:
            if self._token is token:
                self._token = None
                self._name = None
