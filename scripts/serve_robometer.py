#!/usr/bin/env python3
"""Stateless RoboMeter prefix inference, with a content-pinned model identity.

Run in an environment containing RoboMeter/PyTorch, independently of JAX.
No robot connection and no automatic service launch from the training process.
"""
from __future__ import annotations

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import sys
import threading

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hilserl.reward_provider import PROTOCOL, digest


def model_manifest(directory):
    root = Path(directory).resolve()
    files = sorted(p for p in root.rglob("*") if p.is_file()
                   and p.suffix in {".json", ".safetensors", ".bin", ".model", ".txt", ".py"}
                   and not any(part.startswith(".") for part in p.relative_to(root).parts))
    if not any(p.suffix in {".safetensors", ".bin"} for p in files):
        raise ValueError("No model weights found")
    manifest = {}
    for p in files:
        h = hashlib.sha256()
        with p.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                h.update(chunk)
        manifest[str(p.relative_to(root))] = h.hexdigest()
    return dict(model_id=digest(manifest), files=manifest)


def make_server(address, backend, identity):
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, code, value):
            payload = json.dumps(value, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {"error": "Not found"})
            self.respond(200, dict(ready=True, protocol=PROTOCOL, model_id=identity))

        def do_POST(self):
            if self.path != "/score":
                return self.respond(404, {"error": "Not found"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 128 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                with np.load(io.BytesIO(self.rfile.read(size)), allow_pickle=False) as arrays:
                    meta = json.loads(str(arrays["metadata"].item()))
                    if meta["model_id"] != identity or meta["protocol"] != PROTOCOL:
                        raise ValueError("Model identity/protocol mismatch")
                    queries = meta["query_ids"]
                    if (not isinstance(queries, list) or not 1 <= len(queries) <= 64
                            or any(type(q) is not int or q < 0 for q in queries)
                            or len(set(queries)) != len(queries) or not meta["task"].strip()):
                        raise ValueError("Invalid queries/task")
                    samples = []
                    for i, query in enumerate(queries):
                        frames = arrays[f"sample_{i}"]
                        if (frames.ndim != 4 or frames.dtype != np.uint8 or frames.shape[-1] != 3
                                or not 1 <= frames.shape[0] <= min(8, query + 1)
                                or not 1 <= frames.shape[1] <= 2048 or not 1 <= frames.shape[2] <= 2048):
                            raise ValueError("Invalid RGB prefix shape")
                        samples.append(dict(sample_type="progress", trajectory=dict(
                            frames=frames, frames_shape=list(frames.shape), task=meta["task"],
                            id=f"query-{query}", metadata=dict(subsequence_length=len(frames),
                            view_name=meta["image_key"], source=PROTOCOL),
                            video_embeddings=None, text_embedding=None)))
                with lock:
                    outputs = backend.predict_progress_samples(samples)
                if len(outputs) != len(queries):
                    raise ValueError("Wrong number of model outputs")
                scores = []
                for values in outputs:
                    if not values or not np.isfinite(values).all():
                        raise ValueError("Missing/nonfinite model progress")
                    scores.append(float(np.clip(values[-1], 0, 1)))
                self.respond(200, dict(protocol=PROTOCOL, model_id=identity, query_ids=queries, scores=scores))
            except Exception as exc:
                self.respond(400, {"error": f"{type(exc).__name__}: {exc}"})

    return ThreadingHTTPServer(address, Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--robometer-root")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=20008)
    parser.add_argument("--forward-batch-size", type=int, default=4)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    manifest = model_manifest(args.model_path)
    print(json.dumps(manifest), flush=True)
    if args.manifest_only:
        return
    from hilserl.robometer_backend import RoboMeterNativeBackend
    backend = RoboMeterNativeBackend(model_path=args.model_path, device=args.device,
        forward_batch_size=args.forward_batch_size, robometer_root=args.robometer_root)
    # Detect files replaced while loading; the running model is pinned to this manifest.
    if model_manifest(args.model_path) != manifest:
        raise RuntimeError("Model files changed while loading")
    server = make_server((args.host, args.port), backend, manifest["model_id"])
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
