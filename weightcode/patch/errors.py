"""Error type of the patch executor and the collector that gathers validation messages."""

from __future__ import annotations

from collections.abc import Iterable

MAX_MESSAGES = 100


class PatchError(ValueError):
    """A patch was rejected. ``messages`` lists every problem found, in the order found.

    Only problems with the patch itself (format, references, file safety, results, size limits) raise
    ``PatchError``. Caller errors, such as a parent that is not float32 or a malformed expected hash, raise
    ``TypeError`` or a plain ``ValueError``; catch ``PatchError`` first to tell the two apart.
    """

    def __init__(self, messages: str | Iterable[str]) -> None:
        items = [messages] if isinstance(messages, str) else [str(message) for message in messages]
        if not items:
            raise ValueError("PatchError needs at least one message")
        self.messages: list[str] = items
        super().__init__("\n".join(items))

    def __reduce__(self) -> tuple[type[PatchError], tuple[list[str]]]:
        return (type(self), (list(self.messages),))


class MessageLog:
    """Collects ``"<where>: <text>"`` messages, keeping the first ``limit`` and counting the rest."""

    def __init__(self, limit: int = MAX_MESSAGES) -> None:
        self._limit = limit
        self._messages: list[str] = []
        self._omitted = 0

    def add(self, where: str, text: str) -> None:
        """Record one problem found at ``where``."""
        if len(self._messages) < self._limit:
            self._messages.append(f"{where}: {text}")
        else:
            self._omitted += 1

    @property
    def count(self) -> int:
        """Number of problems recorded so far, including omitted ones."""
        return len(self._messages) + self._omitted

    def raise_if_any(self) -> None:
        """Raise ``PatchError`` with every recorded message if any problem was recorded."""
        if not self.count:
            return
        messages = list(self._messages)
        if self._omitted:
            messages.append(f"{self._omitted} further problems were found and not listed")
        raise PatchError(messages)
