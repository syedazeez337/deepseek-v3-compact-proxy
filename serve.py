"""Local playground server for a trained Compact V3 checkpoint.

Streams tokens over Server-Sent Events so generation appears as it is produced.
Standard library only, no new dependencies.

    uv run python serve.py --checkpoint checkpoints/compact_v3_wikitext103_2ep.pt

Then open http://127.0.0.1:8000 in a browser.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import torch
from tokenizers import Tokenizer

from compact_v3.config import config_from_checkpoint
from compact_v3.data import load_tokenizer, resolve_tokenizer_path
from compact_v3.generation import greedy_next_token, sample_next_token
from compact_v3.model import CompactV3Model

UI_PATH = Path(__file__).parent / "ui" / "index.html"
# One GPU, one model: serialise generation so concurrent tabs cannot interleave
# decode steps through the same KV cache.
GENERATION_LOCK = threading.Lock()


class Engine:
    def __init__(self, checkpoint: Path, tokenizer_path: Path | None, device: torch.device) -> None:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
        self.config = config_from_checkpoint(payload["model_config"])
        self.model = CompactV3Model(self.config).to(device)
        self.model.load_state_dict(payload["model"])
        self.model.eval()
        self.device = device
        self.step = payload.get("step")
        self.checkpoint_name = checkpoint.name

        resolved = resolve_tokenizer_path(tokenizer_path, payload)
        self.tokenizer: Tokenizer = load_tokenizer(resolved)
        self.tokenizer_name = str(resolved)

        metrics = payload.get("metrics") or {}
        self.validation_perplexity = metrics.get("validation_perplexity")

    def info(self) -> dict:
        return {
            "checkpoint": self.checkpoint_name,
            "step": self.step,
            "tokenizer": self.tokenizer_name,
            "validation_perplexity": self.validation_perplexity,
            "context_length": self.config.context_length,
            "parameters": sum(p.numel() for p in self.model.parameters()),
            "n_routed_experts": self.config.n_routed_experts,
            "top_k": self.config.top_k,
            "n_layer": self.config.n_layer,
            "d_model": self.config.d_model,
            "device": str(self.device),
        }

    @torch.inference_mode()
    def stream(self, prompt: str, max_new_tokens: int, temperature: float,
               top_k: int | None, top_p: float, do_sample: bool, seed: int | None):
        prompt_ids = self.tokenizer.encode(prompt).ids
        if not prompt_ids:
            yield {"event": "error", "message": "prompt encoded to zero tokens"}
            return

        budget = self.config.context_length - len(prompt_ids) - 1
        if budget <= 0:
            yield {"event": "error",
                   "message": f"prompt is {len(prompt_ids)} tokens; context is {self.config.context_length}"}
            return
        max_new_tokens = min(max_new_tokens, budget)

        generator = None
        if do_sample and seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        tokens = torch.tensor([prompt_ids], device=self.device)
        start = time.perf_counter()
        logits, caches = self.model.prefill(tokens)
        emitted: list[int] = []
        text_so_far = ""

        yield {
            "event": "start",
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": max_new_tokens,
            "context_length": self.config.context_length,
            "checkpoint": self.checkpoint_name,
        }

        for index in range(max_new_tokens):
            next_logits = logits[:, -1]
            if do_sample:
                nxt = sample_next_token(next_logits, temperature, top_k, top_p, generator)
            else:
                nxt = greedy_next_token(next_logits)
            token_id = int(nxt.item())
            emitted.append(token_id)

            # Decode the whole continuation each step and emit the delta, so
            # multi-token characters are never split mid-sequence.
            decoded = self.tokenizer.decode(emitted)
            delta, text_so_far = decoded[len(text_so_far):], decoded
            elapsed = time.perf_counter() - start
            yield {
                "event": "token",
                "text": delta,
                # The id and its own decoded form let the client show real token
                # boundaries, which is the point of a model playground.
                "id": token_id,
                "piece": self.tokenizer.decode([token_id]),
                "index": index + 1,
                "tokens_per_second": (index + 1) / elapsed if elapsed > 0 else 0.0,
            }
            logits, caches = self.model.decode(nxt, caches)

        elapsed = time.perf_counter() - start
        yield {
            "event": "done",
            "generated": len(emitted),
            "seconds": round(elapsed, 3),
            "tokens_per_second": round(len(emitted) / elapsed, 2) if elapsed > 0 else 0.0,
        }


def make_handler(engine: Engine):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # SSE writes one small frame per token. With Nagle enabled each frame
        # waits for an ACK before going out, which measured 14.4 tok/s against
        # the model's own 40 tok/s. Sending immediately removes that stall.
        disable_nagle_algorithm = True

        def log_message(self, *args):  # keep the console readable
            pass

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            route = urlparse(self.path).path
            if route in ("/", "/index.html"):
                if not UI_PATH.exists():
                    self._send(500, b"ui/index.html is missing", "text/plain; charset=utf-8")
                    return
                self._send(200, UI_PATH.read_bytes(), "text/html; charset=utf-8")
            elif route == "/api/info":
                self._send(200, json.dumps(engine.info()).encode(), "application/json")
            else:
                self._send(404, b"not found", "text/plain; charset=utf-8")

        def do_POST(self):
            if urlparse(self.path).path != "/api/generate":
                self._send(404, b"not found", "text/plain; charset=utf-8")
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                request = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._send(400, b"invalid json", "text/plain; charset=utf-8")
                return

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # The stream is finite: one generation then EOF. Without an explicit
            # close the client's reader never completes, which left the UI stuck
            # in its "generating" state after the first request.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            def emit(payload: dict) -> None:
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                self.wfile.flush()

            top_k = request.get("top_k")
            try:
                with GENERATION_LOCK:
                    for chunk in engine.stream(
                        prompt=str(request.get("prompt", "")),
                        max_new_tokens=int(request.get("max_new_tokens", 64)),
                        temperature=float(request.get("temperature", 0.8)),
                        top_k=int(top_k) if top_k else None,
                        top_p=float(request.get("top_p", 1.0)),
                        do_sample=bool(request.get("do_sample", True)),
                        seed=request.get("seed"),
                    ):
                        emit(chunk)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the browser navigated away mid-stream
            except Exception as exc:  # surface model errors in the UI rather than a dead stream
                try:
                    emit({"event": "error", "message": f"{type(exc).__name__}: {exc}"})
                except OSError:
                    pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a Compact V3 checkpoint with a browser playground.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/compact_v3_wikitext103_2ep.pt"))
    parser.add_argument("--tokenizer", type=Path, default=None,
                        help="defaults to the tokenizer recorded in the checkpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    engine = Engine(args.checkpoint, args.tokenizer, torch.device(args.device))
    info = engine.info()
    print(json.dumps(info, indent=2))
    print(f"\n  playground on http://{args.host}:{args.port}\n  ctrl-c to stop\n", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), make_handler(engine))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("stopping")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
