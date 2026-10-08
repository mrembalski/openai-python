from __future__ import annotations

import sys
import time
import logging
from typing import Any, Iterator, AsyncIterator, cast

import httpx2

from ._utils import lru_cache
from ._httpx2 import _loaded_legacy_httpx

log: logging.Logger = logging.getLogger(__name__)

# How long to wait for the bytes that trail a logically complete body before
# giving up and closing the socket. After `data: [DONE]` the only thing left is
# the HTTP/1.1 chunked terminator, which is already in flight when the consumer
# stops reading, so this almost never waits at all.
DRAIN_TIMEOUT = 0.1


def wrap_response_stream(response: Any) -> None:
    """Make closing a partially read streamed response drain its trailing bytes.

    httpcore only returns a connection to the pool when the whole response body
    was parsed. The SDK stops reading at `data: [DONE]`, one chunk before the
    terminator, so without this every streaming request opens a new connection.
    """
    stream = response.stream
    for name, module in _http_modules():
        sync_cls, async_cls = _wrapper_classes(name)
        if isinstance(stream, module.AsyncByteStream):
            response.stream = async_cls(stream, response.request)
            return
        if isinstance(stream, module.SyncByteStream):
            response.stream = sync_cls(stream, response.request)
            return


def _http_modules() -> list[tuple[str, Any]]:
    modules: list[tuple[str, Any]] = [("httpx2", httpx2)]
    legacy = _loaded_legacy_httpx()
    if legacy is not None:
        modules.append(("httpx", cast(Any, legacy)))
    return modules


class _ReadTimeout:
    """Temporarily lower the read timeout httpcore picks up for a fresh body iteration."""

    def __init__(self, request: Any, read_timeout: float) -> None:
        self._request = request
        self._read_timeout = read_timeout
        self._saved: Any = None

    def __enter__(self) -> None:
        self._saved = self._request.extensions.get("timeout")
        self._request.extensions["timeout"] = {**(self._saved or {}), "read": self._read_timeout}

    def __exit__(self, *_: object) -> None:
        if self._saved is None:
            self._request.extensions.pop("timeout", None)
        else:
            self._request.extensions["timeout"] = self._saved


class _SyncDrain:
    _stream: Any
    _request: Any
    _exhausted: bool
    _closing: bool

    def __init__(self, stream: Any, request: Any) -> None:
        self._stream = stream
        self._request = request
        self._exhausted = False
        self._closing = False

    def __iter__(self) -> Iterator[bytes]:
        yield from self._stream
        self._exhausted = True

    def close(self) -> None:
        # A body read to the end left h11 in DONE with an empty buffer; iterating
        # again would block on the socket for the whole drain timeout.
        if not self._exhausted and not self._closing:
            self._closing = True
            deadline = time.monotonic() + DRAIN_TIMEOUT
            try:
                with _ReadTimeout(self._request, DRAIN_TIMEOUT):
                    for _ in self._stream:
                        if time.monotonic() > deadline:
                            break
            except Exception as exc:
                log.debug("Giving up on draining the response body: %s", type(exc).__name__)
        self._stream.close()


class _AsyncDrain:
    _stream: Any
    _request: Any
    _exhausted: bool
    _closing: bool

    def __init__(self, stream: Any, request: Any) -> None:
        self._stream = stream
        self._request = request
        self._exhausted = False
        self._closing = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk
        self._exhausted = True

    async def aclose(self) -> None:
        if not self._exhausted and not self._closing:
            self._closing = True
            deadline = time.monotonic() + DRAIN_TIMEOUT
            try:
                with _ReadTimeout(self._request, DRAIN_TIMEOUT):
                    async for _ in self._stream:
                        if time.monotonic() > deadline:
                            break
            except Exception as exc:
                log.debug("Giving up on draining the response body: %s", type(exc).__name__)
        await self._stream.aclose()


@lru_cache(maxsize=None)
def _wrapper_classes(module_name: str) -> tuple[type[Any], type[Any]]:
    # `Response.close()` type-checks its stream against the ByteStream classes of
    # the module that produced it, so the wrappers must inherit from that module
    # (`httpx2`, or legacy `httpx` for clients built with it).
    module: Any = sys.modules[module_name]
    sync_cls = type("DrainingSyncByteStream", (_SyncDrain, module.SyncByteStream), {})
    async_cls = type("DrainingAsyncByteStream", (_AsyncDrain, module.AsyncByteStream), {})
    return sync_cls, async_cls
