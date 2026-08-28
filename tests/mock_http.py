"""A tiny real HTTP server for testing the adapters.

Adapters are tested against this rather than a mocked-out httpx, because most
of what can go wrong in an API client lives in the parts a mock replaces:
query-string construction, headers, status handling, streaming downloads,
redirects. A stdlib server on localhost exercises all of it and still runs in
milliseconds.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable
from urllib.parse import parse_qs, urlparse


@dataclass
class Request:
    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes = b""

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8")) if self.body else None

    def param(self, name: str, default: str = "") -> str:
        values = self.query.get(name) or []
        return values[0] if values else default


@dataclass
class Response:
    status: int = 200
    body: bytes = b""
    content_type: str = "application/json"

    @classmethod
    def json(cls, payload: Any, status: int = 200) -> "Response":
        return cls(status, json.dumps(payload).encode("utf-8"), "application/json")

    @classmethod
    def binary(cls, data: bytes, content_type: str = "video/mp4") -> "Response":
        return cls(200, data, content_type)


Handler = Callable[[Request], Response]


@dataclass
class MockServer:
    """Serves `handler` on an ephemeral localhost port."""

    handler: Handler
    requests: list[Request] = field(default_factory=list)
    _server: ThreadingHTTPServer | None = None
    _thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        assert self._server is not None, "server not started"
        host, port = self._server.server_address[:2]
        return f"http://127.0.0.1:{port}"

    def paths(self) -> list[str]:
        return [request.path for request in self.requests]

    def count(self, path: str) -> int:
        return sum(1 for request in self.requests if request.path == path)

    def start(self) -> "MockServer":
        outer = self

        class _Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _dispatch(self, method: str) -> None:
                parsed = urlparse(self.path)
                length = int(self.headers.get("Content-Length") or 0)
                request = Request(
                    method=method,
                    path=parsed.path,
                    query=parse_qs(parsed.query),
                    headers={k.lower(): v for k, v in self.headers.items()},
                    body=self.rfile.read(length) if length else b"",
                )
                outer.requests.append(request)

                try:
                    response = outer.handler(request)
                except Exception as exc:  # noqa: BLE001 - surface as a 500
                    response = Response.json({"error": str(exc)}, status=500)

                self.send_response(response.status)
                self.send_header("Content-Type", response.content_type)
                self.send_header("Content-Length", str(len(response.body)))
                self.end_headers()
                if response.body:
                    self.wfile.write(response.body)

            def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
                self._dispatch("GET")

            def do_POST(self):  # noqa: N802
                self._dispatch("POST")

            def log_message(self, *args):  # silence stderr access logs
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> "MockServer":
        return self.start()

    def __exit__(self, *exc_info) -> None:
        self.stop()


def routed(routes: dict[str, Handler], fallback: Response | None = None) -> Handler:
    """Build a handler from a {path: handler} mapping."""
    def dispatch(request: Request) -> Response:
        route = routes.get(request.path)
        if route is None:
            return fallback or Response.json(
                {"error": f"no route for {request.path}"}, status=404
            )
        return route(request)

    return dispatch
