# TorchCodec / transformers import-order repro & workaround

This small folder contains reproducible scripts demonstrating an import-order issue between
`torchcodec` and `transformers` (AST audio feature modules), and a safe workaround that
preloads the system `libavcodec` before loading `torchcodec`'s extension.

Files
- `repro_fail_order.py` — shows the failing import order (torchcodec first).
- `repro_fix_ctypes_preload.py` — preloads `libavcodec` with `ctypes.CDLL(..., RTLD_GLOBAL)` before loading torchcodec.

Usage
1. Inspect which libavcodec is available:

```bash
ldconfig -p | grep avcodec
python -c "import ctypes.util; print(ctypes.util.find_library('avcodec'))"
```

2. Run the example that preloads the library (recommended):

```bash
python scripts/torchcodec_preload/repro_fix_ctypes_preload.py
```

Notes
- Replace the `ctypes.util.find_library('avcodec')` result with a full path if needed (e.g. `/usr/lib/.../libavcodec.so.58`).
- If you cannot change import order in your app, the preload approach or LD_PRELOAD are the most reliable fixes.
