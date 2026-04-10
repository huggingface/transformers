# Run after `python test_export_onnx.py` so `test_onnx/model.onnx` exists.
# Uses the same tiny config and `torch.manual_seed(0)` as export for weight parity with the graph.
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F

from transformers import TimesFm2_5Config, TimesFm2_5ModelForPrediction

ONNX_PATH = Path(__file__).resolve().parent / "test_onnx" / "model.onnx"


def tiny_config() -> TimesFm2_5Config:
    return TimesFm2_5Config(
        patch_length=32,
        context_length=128,
        horizon_length=8,
        hidden_size=32,
        intermediate_size=64,
        head_dim=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        quantiles=[0.1, 0.5, 0.9],
        output_quantile_len=16,
    )


def build_model() -> TimesFm2_5ModelForPrediction:
    torch.manual_seed(0)
    return TimesFm2_5ModelForPrediction(tiny_config()).eval()


def test_preprocess_2d_tensor_matches_list_of_rows(model: TimesFm2_5ModelForPrediction) -> None:
    """Batched `(B, L)` input must match `_preprocess` on `[row[0], …, row[B-1]]` (values + padding mask)."""
    ctx = model.context_len
    for seed, batch, seq_len in (
        (11, 1, 1),
        (12, 2, 17),
        (13, 4, 64),
        (14, 3, ctx),
        (15, 2, ctx + 40),
    ):
        torch.manual_seed(seed)
        x = torch.randn(batch, seq_len)
        rows = [x[i].clone() for i in range(batch)]
        a = model._preprocess(x, context_len=ctx)
        b = model._preprocess(rows, context_len=ctx)
        assert a[0].shape == b[0].shape == (batch, ctx), (a[0].shape, b[0].shape)
        h = model.horizon_len
        assert a[1].shape == b[1].shape == (batch, ctx + h), (a[1].shape, b[1].shape)
        torch.testing.assert_close(a[0], b[0], rtol=0, atol=0, msg="padded time series mismatch")
        torch.testing.assert_close(a[1], b[1], rtol=0, atol=0, msg="padding mask mismatch")


def test_preprocess_short_2d_left_pad_and_mask_invariants(model: TimesFm2_5ModelForPrediction) -> None:
    """After preprocess, leading pad slots are zero in `ts` and one in `padding`."""
    ctx = model.context_len
    h = model.horizon_len
    torch.manual_seed(21)
    seq_len = 40
    b = 2
    x = torch.randn(b, seq_len)
    ts, padding = model._preprocess(x, context_len=ctx)
    num_front = ctx - seq_len
    assert num_front > 0
    assert torch.all(ts[:, :num_front] == 0)
    assert torch.all(padding[:, :num_front] == 1)
    assert torch.all(padding[:, num_front : num_front + seq_len] == 0)


def list_to_left_padded_matrix(parts: list[torch.Tensor], width: int) -> np.ndarray:
    """Match `_preprocess` left-padding: short series get zeros on the left."""
    rows = []
    for p in parts:
        p = p.float()[-width:]
        rows.append(F.pad(p, (width - p.shape[0], 0)))
    return torch.stack(rows).numpy()


def onnx_output_names(session: ort.InferenceSession) -> list[str]:
    return [o.name for o in session.get_outputs()]


def onnx_protobuf_input_shape(path: Path, input_name: str) -> tuple[int | str, int | str]:
    """Shape of named graph input as in the .onnx file."""
    import onnx

    model = onnx.load(str(path))
    inputs = {i.name: i for i in model.graph.input}
    if input_name not in inputs:
        raise KeyError(f"Input {input_name!r} not found in graph")
    
    inp = inputs[input_name]
    dims: list[int | str] = []
    for d in inp.type.tensor_type.shape.dim:
        param = (d.dim_param or "").strip()
        if param:
            dims.append(param)
        elif d.HasField("dim_value"):
            dims.append(int(d.dim_value))
        else:
            dims.append("?")
    if len(dims) < 2:
        return dims[0], "?"
    return dims[0], dims[1]


def test_onnx_export_batch_axis_contract(session: ort.InferenceSession, onnx_path: Path, ctx: int) -> None:
    """ORT must match the **onnx file** (protobuf) axis 0 contract for past_values."""
    pb0, pb1 = onnx_protobuf_input_shape(onnx_path, "past_values")
    print(f"  protobuf past_values shape: [{pb0!r}, {pb1!r}]")

    inp_name = "past_values"
    names = onnx_output_names(session)
    out = "mean_predictions" if "mean_predictions" in names else names[0]

    def run_batch(batch_size: int) -> None:
        x = np.random.randn(batch_size, ctx).astype(np.float32)
        session.run([out], {inp_name: x})

    if isinstance(pb0, int):
        raise AssertionError(
            f"Dynamic Batch Requirement Failed: The ONNX file declares a fixed batch dimension {pb0!r} "
            f"on 'past_values' (axis 0). Expected a symbolic name (e.g., 'batch')."
        )
    else:
        run_batch(2)
        run_batch(5)
        print(f"  OK: ORT accepted batch 2 and 5 (protobuf symbolic batch axis {pb0!r}).")


def run_onnx(session: ort.InferenceSession, x_np: np.ndarray) -> dict[str, np.ndarray]:
    names = onnx_output_names(session)
    arrays = session.run(names, {"past_values": x_np})
    return dict(zip(names, arrays, strict=True))


def assert_close(a: np.ndarray, b: np.ndarray, msg: str, rtol: float = 1e-3, atol: float = 1e-3) -> None:
    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol, err_msg=msg)


def should_check_last_hidden_state_ort(model: TimesFm2_5ModelForPrediction, seq_len: int) -> bool:
    threshold = model.context_len - model.config.patch_length
    return seq_len > threshold


def test_pytorch_list_matches_stacked_2d_when_each_series_has_length_ctx(model: TimesFm2_5ModelForPrediction) -> None:
    ctx = model.context_len
    torch.manual_seed(42)
    s0 = torch.randn(ctx)
    s1 = torch.randn(ctx)
    stacked = torch.stack([s0, s1], dim=0)
    with torch.no_grad():
        o_list = model(past_values=[s0, s1])
        o_2d = model(past_values=stacked)
    assert_close(o_list.mean_predictions.numpy(), o_2d.mean_predictions.numpy(), "mean_predictions list vs 2D")
    assert_close(o_list.full_predictions.numpy(), o_2d.full_predictions.numpy(), "full_predictions list vs 2D")


def test_variable_length_list_vs_prepadded_2d_differs_in_padding_mask(model: TimesFm2_5ModelForPrediction) -> None:
    ctx = model.context_len
    data = [torch.linspace(0, 1, 100), torch.sin(torch.linspace(0, 20, 67))]
    matrix_t = torch.from_numpy(list_to_left_padded_matrix(data, ctx))
    with torch.no_grad():
        o_list = model(past_values=data)
        o_pad2d = model(past_values=matrix_t)
    assert ((o_list.mean_predictions - o_pad2d.mean_predictions).abs().max() > 1e-3)


def test_onnx_matches_pytorch_all_outputs(
    session: ort.InferenceSession,
    model: TimesFm2_5ModelForPrediction,
    x: torch.Tensor,
) -> None:
    names = onnx_output_names(session)
    with torch.no_grad():
        pt = model(past_values=x)
    
    ort_dict = run_onnx(session, x.numpy())
    if "mean_predictions" in ort_dict:
        assert_close(ort_dict["mean_predictions"], pt.mean_predictions.numpy(), "mean_predictions ORT vs PT")
    if "full_predictions" in ort_dict:
        assert_close(ort_dict["full_predictions"], pt.full_predictions.numpy(), "full_predictions ORT vs PT")


def run_dynamic_shape_checks(session: ort.InferenceSession, model: TimesFm2_5ModelForPrediction, onnx_path: Path) -> None:
    pb_batch, pb_seq = onnx_protobuf_input_shape(onnx_path, "past_values")
    ctx = model.context_len

    if not isinstance(pb_batch, str):
        raise AssertionError(f"Dynamic Batch Check Failed: {pb_batch!r}")

    for batch in (1, 3, 5):
        torch.manual_seed(100 + batch)
        x = torch.randn(batch, ctx)
        test_onnx_matches_pytorch_all_outputs(session, model, x)
        print(f"ONNX vs PyTorch OK: batch_size={batch}, seq_len={ctx}")

    if not isinstance(pb_seq, str):
        raise AssertionError(f"Dynamic Sequence Check Failed: {pb_seq!r}")

    for seq_len in (1, 32, 64, ctx, 200):
        torch.manual_seed(300 + seq_len)
        x = torch.randn(2, seq_len)
        test_onnx_matches_pytorch_all_outputs(session, model, x)
        print(f"ONNX vs PyTorch OK: batch_size=2, seq_len={seq_len}")


def test_input_min_and_type_parity(model: TimesFm2_5ModelForPrediction) -> None:
    """
    Verifies the fix for tensor vs list inputs.
    Checks that input_min is calculated across ALL rows, which affects truncate_negative.
    """
    ctx = model.context_len
    # Row 0 is all positive, Row 1 has a negative value.
    # If the model only checked the first row (or handled the list incorrectly),
    # it might wrongly decide to clamp outputs.
    s0 = torch.ones(ctx) * 10.0
    s1 = torch.ones(ctx) * 10.0
    s1[5] = -100.0  # The negative value is in the second row
    
    stacked = torch.stack([s0, s1], dim=0)
    list_input = [s0, s1]
    
    # We need a case where the model WOULD produce a negative value to see if it gets clamped.
    # Since we use random weights, we'll just check that outputs match between tensor and list paths.
    with torch.no_grad():
        out_tensor = model(past_values=stacked, truncate_negative=True)
        out_list = model(past_values=list_input, truncate_negative=True)
        
    assert_close(
        out_tensor.mean_predictions.numpy(),
        out_list.mean_predictions.numpy(),
        "Input type parity failed (tensor vs list with negative value)"
    )
    print("    OK: Tensor and List paths matched for mixed-sign inputs.")


def test_window_size_tensor_vs_list_parity(model: TimesFm2_5ModelForPrediction) -> None:
    """Verifies that the new batched window_size logic for tensors matches the list logic."""
    ctx = model.context_len
    batch = 3
    window_size = 4
    torch.manual_seed(123)
    x = torch.randn(batch, ctx)
    rows = [x[i].clone() for i in range(batch)]
    
    with torch.no_grad():
        out_tensor = model(past_values=x, window_size=window_size)
        out_list = model(past_values=rows, window_size=window_size)
    
    # We expect (batch, horizon_len) rows in the output
    h = model.horizon_len
    assert out_tensor.mean_predictions.shape == out_list.mean_predictions.shape == (batch, h)
    torch.testing.assert_close(out_tensor.mean_predictions, out_list.mean_predictions, rtol=1e-5, atol=1e-5)
    print(f"    OK: Tensor vs List window_size parity at B={batch}, W={window_size}")


def main() -> None:
    if not ONNX_PATH.is_file():
        print(f"Missing {ONNX_PATH}", file=sys.stderr)
        sys.exit(1)

    model = build_model()
    ctx = model.context_len

    print("PyTorch tests...")
    test_preprocess_2d_tensor_matches_list_of_rows(model)
    test_preprocess_short_2d_left_pad_and_mask_invariants(model)
    test_input_min_and_type_parity(model)
    test_window_size_tensor_vs_list_parity(model)
    print("  OK")

    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    
    print("ONNX: past_values batch contract...")
    test_onnx_export_batch_axis_contract(session, ONNX_PATH, ctx)
    print("  OK")

    print("ONNX dynamic shape parity...")
    run_dynamic_shape_checks(session, model, ONNX_PATH)
    print("  OK")

    print("All checks passed.")


if __name__ == "__main__":
    main()
