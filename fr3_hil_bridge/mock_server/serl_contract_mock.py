#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any


STATE = {
    "pose": [0.45, 0.0, 0.30, 0.0, 0.0, 0.0, 1.0],
    "vel": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "force": [0.0, 0.0, 0.0],
    "torque": [0.0, 0.0, 0.0],
    "q": [0.0, -0.78, 0.0, -2.35, 0.0, 1.57, 0.78],
    "dq": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "jacobian": [[0.0] * 7 for _ in range(6)],
    "gripper_pos": 0.08,
}

READ_ENDPOINTS = {
    "/getstate": STATE,
    "/getpos": {"pose": STATE["pose"]},
    "/getvel": {"vel": STATE["vel"]},
    "/getforce": {"force": STATE["force"]},
    "/gettorque": {"torque": STATE["torque"]},
    "/getq": {"q": STATE["q"]},
    "/getdq": {"dq": STATE["dq"]},
    "/getjacobian": {"jacobian": STATE["jacobian"]},
    "/get_gripper": {"gripper": STATE["gripper_pos"]},
}

COMMAND_ENDPOINTS = {
    "/pose",
    "/jointreset",
    "/activate_gripper",
    "/reset_gripper",
    "/open_gripper",
    "/close_gripper",
    "/close_gripper_slow",
    "/move_gripper",
    "/set_load",
    "/startimp",
    "/stopimp",
    "/clearerr",
    "/update_param",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "SERLContractMock/0.1"

    def do_POST(self) -> None:
        _ = self.rfile.read(int(self.headers.get("content-length", "0") or "0"))
        if self.path in READ_ENDPOINTS:
            self._json(200, READ_ENDPOINTS[self.path])
            return
        if self.path in COMMAND_ENDPOINTS:
            self._json(409, {"error": "NO_MOTION_MOCK_REJECTED", "endpoint": self.path})
            return
        self._json(404, {"error": "UNKNOWN_ENDPOINT", "endpoint": self.path})

    def do_GET(self) -> None:
        if self.path == "/healthz":
            self._json(200, {"ok": True, "mode": "no-motion"})
            return
        self._json(404, {"error": "UNKNOWN_ENDPOINT", "endpoint": self.path})

    def log_message(self, fmt: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), fmt % args), flush=True)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5017)
    args = parser.parse_args()
    if args.host not in {"127.0.0.1", "localhost"}:
        raise SystemExit("mock server must bind localhost only")
    HTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
