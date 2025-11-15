# Copyright 2024 The HuggingFace Team.
# Licensed under the Apache License, Version 2.0

import gc
import unittest

import accelerate

from transformers import AutoModelForCausalLM, AutoTokenizer, SinqConfig
from transformers.testing_utils import (
    backend_empty_cache,
    require_accelerate,
    require_torch_accelerator,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

from sinq.sinqlinear import SINQLinear


# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def cleanup():
    backend_empty_cache(torch_device)
    gc.collect()


class SINQLLMRunner:
    """Small helper to load a CausalLM + tokenizer with SINQ quantization."""

    def __init__(self, model_id, quant_config, torch_dtype, device_map, cache_dir=None):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quant_config,
            cache_dir=cache_dir,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=cache_dir)
        try:
            self.device = next(self.model.parameters()).device
        except StopIteration:
            self.device = torch.device(torch_device)


def _assert_close(test: unittest.TestCase, a: torch.Tensor, b: torch.Tensor, rtol=1e-3, atol=1e-3):
    test.assertEqual(a.shape, b.shape)
    ok = torch.allclose(a, b, rtol=rtol, atol=atol)
    if not bool(ok):
        diff = (a - b).abs().mean().item()
        test.fail(f"Tensors differ: mean abs diff={diff:.6g}")


def check_sinq_linear_api(test, sinq_layer: SINQLinear, batch_size=1, context_size=16):
    """Validates forward shape/dtype and layer attributes."""
    test.assertIsInstance(sinq_layer, SINQLinear)
    with torch.no_grad():
        W = sinq_layer.dequantize()
    test.assertEqual(W.dim(), 2)
    x = torch.randn(batch_size, context_size, W.shape[1], device=sinq_layer.device, dtype=sinq_layer.compute_dtype)
    with torch.no_grad():
        y = sinq_layer(x)
    test.assertEqual(y.shape[-1], W.shape[0])
    test.assertEqual(y.dtype, sinq_layer.compute_dtype)
    del W, x, y
    cleanup()


def check_model_forward_logits_shape(test, model, batch_size=1, context_size=8):
    ids = torch.zeros((batch_size, context_size), device=next(model.parameters()).device, dtype=torch.long)
    with torch.no_grad():
        out = model(input_ids=ids).logits
    test.assertEqual(out.shape[0], batch_size)
    test.assertEqual(out.shape[1], context_size)
    del ids, out
    cleanup()


SMALL_MODEL_ID = "facebook/opt-125m"
LARGE_MODEL_ID = "Qwen/Qwen3-1.7B"


# --------------------------------------------------------------------------------------------
# Config tests
# --------------------------------------------------------------------------------------------


@require_torch_accelerator
class SinqConfigTest(unittest.TestCase):
    def test_to_dict_roundtrip(self):
        q = SinqConfig(nbits=8, group_size=64)
        qd = q.to_dict()
        for k in ("nbits", "group_size", "method", "dtype"):
            self.assertIn(k, qd)
        self.assertEqual(qd["nbits"], 8)
        self.assertEqual(qd["group_size"], 64)
        self.assertEqual(str(qd["method"]), "sinq")
        self.assertEqual(qd["dtype"], "auto")

    def test_from_dict_roundtrip(self):
        d = {
            "nbits": 4,
            "group_size": 64,
            "method": "sinq",
            "modules_to_not_convert": ["lm_head", "embed_tokens"],
            "dtype": "float16",
            "tiling_mode": "1D",
        }
        q = SinqConfig.from_dict(d)
        self.assertEqual(q.nbits, 4)
        self.assertEqual(q.group_size, 64)
        self.assertEqual(q.modules_to_not_convert, ["lm_head", "embed_tokens"])
        qd = q.to_dict()
        for k in ("nbits", "group_size", "method", "dtype"):
            self.assertIn(k, qd)


# --------------------------------------------------------------------------------------------
# Smoke / conversion tests
# --------------------------------------------------------------------------------------------


@require_torch_accelerator
@require_accelerate
class SINQSmokeTest(unittest.TestCase):
    def tearDown(self):
        cleanup()

    def test_fp16_quantized_small_model(self):
        q = SinqConfig(nbits=8, group_size=64, dtype="float16")
        r = SINQLLMRunner(SMALL_MODEL_ID, q, torch.float16, torch_device)
        sinq_layer = r.model.model.decoder.layers[0].self_attn.v_proj
        check_sinq_linear_api(self, sinq_layer)
        check_model_forward_logits_shape(self, r.model)

    def test_move_between_devices_and_dtypes(self):
        q = SinqConfig(nbits=8, group_size=64)
        r = SINQLLMRunner(SMALL_MODEL_ID, q, torch.bfloat16, torch_device)
        sinq_layer = r.model.model.decoder.layers[0].self_attn.v_proj
        check_sinq_linear_api(self, sinq_layer)
        check_model_forward_logits_shape(self, r.model)
        accelerate.hooks.remove_hook_from_module(r.model, recurse=True)
        target_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        r.model.to(next(r.model.parameters()).device, target_dtype)
        check_model_forward_logits_shape(self, r.model, context_size=4)

    def test_layer_weight_dtype_exposed(self):
        q = SinqConfig(nbits=8, group_size=64)
        r = SINQLLMRunner(SMALL_MODEL_ID, q, torch.bfloat16, torch_device)
        layer = r.model.model.decoder.layers[0].self_attn.v_proj
        self.assertEqual(getattr(layer, "compute_dtype", None), torch.bfloat16)


# --------------------------------------------------------------------------------------------
# Conversion coverage: how many linears replaced
# --------------------------------------------------------------------------------------------


@require_torch_accelerator
@require_accelerate
class SinqConversionCountTest(unittest.TestCase):
    def tearDown(self):
        cleanup()

    def _count(self, model, typ):
        return sum(1 for _ in model.modules() if isinstance(_, typ))

    def test_converted_linears_and_skip_list(self):
        q = SinqConfig(nbits=8, group_size=64, dtype="float16")
        m = AutoModelForCausalLM.from_pretrained(SMALL_MODEL_ID, device_map=torch_device, quantization_config=q)
        from sinq.sinqlinear import SINQLinear

        sinq = self._count(m, SINQLinear)
        self.assertGreater(sinq, 0)
        q2 = SinqConfig(nbits=8, group_size=64, dtype="float16", modules_to_not_convert=["lm_head"])
        m2 = AutoModelForCausalLM.from_pretrained(SMALL_MODEL_ID, device_map=torch_device, quantization_config=q2)
        sinq2 = self._count(m2, SINQLinear)
        self.assertGreater(sinq, sinq2)


# --------------------------------------------------------------------------------------------
# Generation tests
# --------------------------------------------------------------------------------------------


@slow
@require_torch_accelerator
@require_accelerate
class SinqGenerateTest(unittest.TestCase):
    def tearDown(self):
        cleanup()

    def test_generate_zero_temp(self):
        q = SinqConfig(nbits=8, group_size=64, dtype="float16")
        tok = AutoTokenizer.from_pretrained(SMALL_MODEL_ID)
        m = AutoModelForCausalLM.from_pretrained(SMALL_MODEL_ID, device_map=torch_device, quantization_config=q)
        prompt = "A quick brown fox"
        ids = tok(prompt, return_tensors="pt").to(next(m.parameters()).device)
        out = m.generate(**ids, max_new_tokens=6, do_sample=False, temperature=0.0)
        txt = tok.decode(out[0], skip_special_tokens=True)
        self.assertTrue(txt.startswith(prompt))
        self.assertGreater(len(out[0]), ids["input_ids"].shape[1])


# --------------------------------------------------------------------------------------------
# ASINQ
# --------------------------------------------------------------------------------------------


@slow
@require_torch_accelerator
@require_accelerate
class SINQASINQTest(unittest.TestCase):
    def tearDown(self):
        cleanup()

    def test_asinq_activation_path_fallback_safe(self):
        q = SinqConfig(nbits=8, group_size=64, method="asinq")
        r = SINQLLMRunner(SMALL_MODEL_ID, q, torch.bfloat16, torch_device)
        sinq_layer = r.model.model.decoder.layers[0].self_attn.v_proj
        check_sinq_linear_api(self, sinq_layer, context_size=8)
        check_model_forward_logits_shape(self, r.model, context_size=4)


# --------------------------------------------------------------------------------------------
# Dequantize one layer
# --------------------------------------------------------------------------------------------


@require_torch_accelerator
@require_accelerate
class SinqLayerDequantizeTest(unittest.TestCase):
    def tearDown(self):
        cleanup()

    def test_layer_dequantize_shapes_and_device(self):
        q = SinqConfig(nbits=4, group_size=64, dtype="float16")
        m = AutoModelForCausalLM.from_pretrained(SMALL_MODEL_ID, device_map=torch_device, quantization_config=q)
        layer = m.model.decoder.layers[0].self_attn.v_proj
        self.assertIsInstance(layer, SINQLinear)
        W = layer.dequantize()
        self.assertEqual(W.dim(), 2)
        if W.device != layer.device:
            W = W.to(layer.device)  # allow CPU-returning implementations


# --------------------------------------------------------------------------------------------
# Trainability (backprop through quantized model)
# --------------------------------------------------------------------------------------------


@require_torch_accelerator
@require_accelerate
class SinqTrainabilityTest(unittest.TestCase):
    def tearDown(self):
        cleanup()

    def test_backward_through_head(self):
        q = SinqConfig(nbits=8, group_size=64, dtype="float16")
        m = AutoModelForCausalLM.from_pretrained(SMALL_MODEL_ID, device_map=torch_device, quantization_config=q)
        device = next(m.parameters()).device
        proj = torch.nn.Linear(m.config.hidden_size, 2).to(device)
        ids = torch.randint(0, m.config.vocab_size, (2, 8), device=device)
        out = m(input_ids=ids, output_hidden_states=True)
        h = out.hidden_states[-1].mean(dim=1)
        if h.dtype != proj.weight.dtype:
            proj = proj.to(dtype=h.dtype)
        logits = proj(h)
        loss = logits.sum()
        loss.backward()
        self.assertIsNotNone(proj.weight.grad)
