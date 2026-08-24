# Verification

Drafting reads the code; verification watches the product. One checklist per cluster of
documents; each item is a single observable claim with setup, steps, an expected result, a
priority, and the terminal it needs. Run the items, record `pass`, `fail`, or `blocked` in the
Result column (with the commit you ran against), and file every failure in
[../bug-triage.md](../bug-triage.md) with the item's ID. A document moves from `drafted` to
`verified` in the README coverage table only when every P1 and P2 item for it has passed or been
filed.

Two ways to run:

## A. Real server

```bash
pip install "transformers[serving]"
transformers serve                       # shell 1
transformers chat HuggingFaceTB/SmolLM3-3B   # shell 2, or any small model
```

Use this for anything about feel, timing with a real model, and the emulator matrix items. Run
P1 items in at least two emulators (one macOS, one Linux or Windows Terminal).

## B. Scripted stub server (deterministic)

For exact terminal sizes, interrupts, and reproducible pacing, drive the real client against a
stub OpenAI-compatible server. Save as `stub_server.py`, run `python stub_server.py 8123`, then
`transformers chat test/model http://127.0.0.1:8123/v1` (a non-default port skips the health
check by design). Asking anything containing `long` streams a 40-row reply at ~20 ms per chunk;
anything else streams a short one; both end with a usage count so the stats line appears.

```python
import json, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8123

def sse(h, payload):
    h.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n"); h.wfile.flush()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or b"{}")
        self.send_response(200); self.send_header("Content-Type", "text/event-stream"); self.end_headers()
        if self.path == "/load_model":
            sse(self, {"status": "loading", "stage": "config"})
            sse(self, {"status": "ready", "cached": True}); return
        user = next((m["content"] for m in reversed(body.get("messages", [])) if m["role"] == "user"), "")
        def chunk(text, finish=None, usage=None):
            p = {"id": "c", "object": "chat.completion.chunk", "created": 0, "model": "m",
                 "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": finish}]}
            if usage: p["usage"] = usage
            sse(self, p)
        text = ("".join(f"ROW{i:02d} lorem ipsum dolor sit amet\n" for i in range(1, 41))
                if "long" in user else "Hello there, short answer.")
        for i in range(0, len(text), 8):
            chunk(text[i:i + 8]); time.sleep(0.02)
        chunk("", "stop", {"prompt_tokens": 3, "completion_tokens": 42, "total_tokens": 45})
        self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()

ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
```

To pin the terminal size, run the chat inside a fixed-size pty (the pty test in
`tests/cli/test_chat_display.py` shows the pattern) or just resize your terminal to 24×80.

## Machine-verified items

Items marked `pass (machine, <commit>)` were verified by automation on that commit: the unit and
pty tests in `tests/cli/test_chat_display.py`, or a scripted pty run of the full client against
the stub server with the output replayed through a terminal emulator. They count as passes for
coverage, but the emulator-matrix items (marked *hand*) still need human eyes — automation ran
one emulated terminal, not the ecosystem.
