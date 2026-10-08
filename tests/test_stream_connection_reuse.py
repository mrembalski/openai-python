from __future__ import annotations

import os
import json
import time
import socket
import importlib
import threading
import socketserver
from typing import Any, Iterator, AsyncIterator, cast
from typing_extensions import override

import httpx2
import pytest

from openai import OpenAI, AsyncOpenAI
from openai._connection_reuse import wrap_response_stream

CHUNK = {
    "id": "chatcmpl-1",
    "object": "chat.completion.chunk",
    "created": 0,
    "model": "synthetic",
    "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
}


class SSEServer(socketserver.ThreadingTCPServer):
    """Keep-alive HTTP/1.1 server that answers every POST with a chunked SSE stream.

    Records the client port of every request so tests can tell connections apart.
    With `stall=True` it sends one event and then holds the response open.
    """

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, *, stall: bool = False) -> None:
        super().__init__(("127.0.0.1", 0), _Handler)
        self.stall = stall
        self.ports: list[int] = []
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    @override
    def __enter__(self) -> SSEServer:
        self._thread.start()
        return self

    @override
    def __exit__(self, *_: object) -> None:
        self.shutdown()
        self.server_close()
        self._thread.join()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}/v1"

    def record(self, port: int) -> None:
        with self._lock:
            self.ports.append(port)


class _Handler(socketserver.StreamRequestHandler):
    @property
    def sse_server(self) -> SSEServer:
        return cast(SSEServer, self.server)

    @override
    def handle(self) -> None:
        self.connection.settimeout(5)
        while True:
            try:
                head = b""
                while b"\r\n\r\n" not in head:
                    data = self.rfile.readline()
                    if not data:
                        return
                    head += data
            except (socket.timeout, ConnectionError):
                return
            headers = dict(line.decode().split(":", 1) for line in head.split(b"\r\n")[1:] if b":" in line)
            length = int(headers.get("content-length", headers.get("Content-Length", "0")).strip())
            self.rfile.read(length)
            self.sse_server.record(self.client_address[1])
            self.wfile.write(
                b"HTTP/1.1 200 OK\r\ncontent-type: text/event-stream\r\ntransfer-encoding: chunked\r\n\r\n"
            )
            self._event(json.dumps(CHUNK))
            if self.sse_server.stall:
                try:
                    self.rfile.read(1)  # wait for the client to go away
                except (socket.timeout, ConnectionError):
                    pass
                return
            self._event(json.dumps(CHUNK))
            self._event("[DONE]")
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()

    def _event(self, data: str) -> None:
        body = f"data: {data}\n\n".encode()
        self.wfile.write(f"{len(body):x}\r\n".encode() + body + b"\r\n")
        self.wfile.flush()


@pytest.fixture(
    params=[
        "httpx2",
        pytest.param(
            "httpx",
            marks=pytest.mark.skipif(
                os.environ.get("OPENAI_TEST_LEGACY_HTTPX") != "1", reason="requires the legacy HTTPX compatibility lane"
            ),
        ),
    ]
)
def http_module(request: pytest.FixtureRequest) -> Any:
    return importlib.import_module(request.param)


@pytest.fixture
def server() -> Iterator[SSEServer]:
    with SSEServer() as srv:
        yield srv


@pytest.fixture
def stalling_server() -> Iterator[SSEServer]:
    with SSEServer(stall=True) as srv:
        yield srv


def _sync_client(http_module: Any, base_url: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key="k", http_client=http_module.Client(trust_env=False))


def _async_client(http_module: Any, base_url: str) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=base_url, api_key="k", http_client=http_module.AsyncClient(trust_env=False))


def test_consumed_streams_reuse_the_connection(http_module: Any, server: SSEServer) -> None:
    with _sync_client(http_module, server.base_url) as client:
        for _ in range(3):
            chunks = list(client.chat.completions.create(model="synthetic", messages=[], stream=True))
            assert len(chunks) == 2
    assert len(server.ports) == 3
    assert len(set(server.ports)) == 1


async def test_consumed_async_streams_reuse_the_connection(http_module: Any, server: SSEServer) -> None:
    async with _async_client(http_module, server.base_url) as client:
        for _ in range(3):
            stream = await client.chat.completions.create(model="synthetic", messages=[], stream=True)
            chunks = [chunk async for chunk in stream]
            assert len(chunks) == 2
    assert len(server.ports) == 3
    assert len(set(server.ports)) == 1


def test_abandoned_stream_closes_promptly(http_module: Any, stalling_server: SSEServer) -> None:
    with _sync_client(http_module, stalling_server.base_url) as client:
        for attempt in range(2):
            stream = client.chat.completions.create(model="synthetic", messages=[], stream=True)
            for _ in stream:
                break
            started = time.monotonic()
            stream.close()
            assert time.monotonic() - started < 1, attempt
    assert len(set(stalling_server.ports)) == 2


async def test_abandoned_async_stream_closes_promptly(http_module: Any, stalling_server: SSEServer) -> None:
    async with _async_client(http_module, stalling_server.base_url) as client:
        for attempt in range(2):
            stream = await client.chat.completions.create(model="synthetic", messages=[], stream=True)
            async for _ in stream:
                break
            started = time.monotonic()
            await stream.close()
            assert time.monotonic() - started < 1, attempt
    assert len(set(stalling_server.ports)) == 2


class _CountingSyncStream(httpx2.SyncByteStream):
    def __init__(self) -> None:
        self.iterations = 0
        self.closed = False

    @override
    def __iter__(self) -> Iterator[bytes]:
        self.iterations += 1
        yield b"data: 1\n\n"
        yield b"data: 2\n\n"

    @override
    def close(self) -> None:
        self.closed = True


class _CountingAsyncStream(httpx2.AsyncByteStream):
    def __init__(self) -> None:
        self.iterations = 0
        self.closed = False

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        self.iterations += 1
        yield b"data: 1\n\n"
        yield b"data: 2\n\n"

    @override
    async def aclose(self) -> None:
        self.closed = True


def _wrapped_sync_response(stream: _CountingSyncStream) -> httpx2.Response:
    request = httpx2.Request("GET", "http://example.test/", extensions={"timeout": {"read": 5.0}})
    response = httpx2.Response(200, stream=stream, request=request)
    wrap_response_stream(response)
    return response


def _wrapped_async_response(stream: _CountingAsyncStream) -> httpx2.Response:
    request = httpx2.Request("GET", "http://example.test/", extensions={"timeout": {"read": 5.0}})
    response = httpx2.Response(200, stream=stream, request=request)
    wrap_response_stream(response)
    return response


def test_fully_read_body_is_not_iterated_again_on_close() -> None:
    stream = _CountingSyncStream()
    response = _wrapped_sync_response(stream)
    response.read()
    response.close()
    assert stream.iterations == 1
    assert stream.closed
    assert response.request.extensions["timeout"] == {"read": 5.0}


async def test_fully_read_async_body_is_not_iterated_again_on_close() -> None:
    stream = _CountingAsyncStream()
    response = _wrapped_async_response(stream)
    await response.aread()
    await response.aclose()
    assert stream.iterations == 1
    assert stream.closed
    assert response.request.extensions["timeout"] == {"read": 5.0}


def test_partially_read_body_is_drained_once_on_close() -> None:
    stream = _CountingSyncStream()
    response = _wrapped_sync_response(stream)
    next(response.iter_bytes())
    response.close()
    response.close()
    assert stream.iterations == 2
    assert stream.closed
    assert response.request.extensions["timeout"] == {"read": 5.0}


async def test_partially_read_async_body_is_drained_once_on_close() -> None:
    stream = _CountingAsyncStream()
    response = _wrapped_async_response(stream)
    await response.aiter_bytes().__anext__()
    await response.aclose()
    await response.aclose()
    assert stream.iterations == 2
    assert stream.closed
    assert response.request.extensions["timeout"] == {"read": 5.0}
