# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import gc
import json
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from parameterized import parameterized

from transformers import EntroPackConfig, LlamaConfig
from transformers.testing_utils import (
    backend_empty_cache,
    require_peft,
    require_torch,
    require_torch_gpu,
    torch_device,
)
from transformers.utils import is_entropack_available, is_torch_available


if is_torch_available():
    import torch

    from transformers import LlamaForCausalLM

if is_entropack_available():
    import entropack

    from transformers.integrations.entropack import is_entropack_linear


require_entropack = unittest.skipUnless(is_entropack_available(), "test requires entropack")


def tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=False,
    )


def compressed_layers(model) -> list:
    return [name for name, module in model.named_modules() if is_entropack_linear(module)]


def disk_tensors(directory: str) -> dict:
    from safetensors.torch import load_file

    merged = {}
    for shard in os.listdir(directory):
        if shard.endswith(".safetensors"):
            merged.update(load_file(os.path.join(directory, shard)))
    return merged


class EntroPackConfigTest(unittest.TestCase):
    def test_to_dict_round_trip(self):
        from transformers.quantizers.auto import AutoQuantizationConfig

        lossy_config = {"row_rdo_iterations": 1}
        config = EntroPackConfig(
            target_bpp=3.5,
            linear_type="int8",
            modules_to_not_convert=["gate_proj"],
            entropack_lossy_config=lossy_config,
        )
        serialized = json.loads(config.to_json_string())
        self.assertEqual(serialized["entropack_lossy_config"], lossy_config)
        restored = AutoQuantizationConfig.from_dict(serialized)
        self.assertIsInstance(restored, EntroPackConfig)
        self.assertEqual(restored.to_dict(), config.to_dict())

    def test_invalid_values_are_rejected(self):
        for kwargs in ({"target_bpp": 100.0}, {"linear_type": "fp4"}, {"linear_type": "fp8", "target_bpp": 8.0}):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                EntroPackConfig(**kwargs)


@require_torch
@require_entropack
class EntroPackQuantizerTest(unittest.TestCase):
    def test_tensor_parallelism_is_rejected_while_preparing_the_model(self):
        from transformers.quantizers.quantizer_entropack import EntroPackHfQuantizer

        config = tiny_config()
        config.distributed_config = SimpleNamespace(tp_size=2)
        with torch.device("meta"):
            model = LlamaForCausalLM(config)
        quantizer = EntroPackHfQuantizer(EntroPackConfig())
        with self.assertRaisesRegex(ValueError, "tensor parallelism"):
            quantizer.preprocess_model(model=model, dtype=torch.bfloat16)

    def test_compute_module_sizes(self):
        from transformers.integrations.accelerate import compute_module_sizes
        from transformers.quantizers.quantizer_entropack import EntroPackHfQuantizer

        for linear_type in ("default", "fp8", "int8"):
            for pre_quantized in (False, True):
                with self.subTest(linear_type=linear_type, pre_quantized=pre_quantized):
                    with torch.device("meta"):
                        model = LlamaForCausalLM(tiny_config()).to(torch.bfloat16)
                    quantizer = EntroPackHfQuantizer(
                        EntroPackConfig(target_bpp=3.5, linear_type=linear_type), pre_quantized=pre_quantized
                    )
                    quantizer.preprocess_model(model)
                    sizes, _ = compute_module_sizes(model, quantizer, only_modules=False)
                    layers = compressed_layers(model)
                    self.assertTrue(layers)
                    for name in layers:
                        module = model.get_submodule(name)
                        self.assertEqual(tuple(module.weight.shape), (module.out_features, module.in_features))
                        self.assertTrue(module.weight.is_meta)
                        self.assertEqual(sizes[f"{name}.weight"], module.weight.numel() * 3.5 / 8)
                        for buffer_name in module.state_buffer_names:
                            buffer = getattr(module, buffer_name)
                            self.assertEqual(sizes[f"{name}.{buffer_name}"], buffer.numel() * buffer.element_size())
                    self.assertEqual(sizes["model.embed_tokens.weight"], model.model.embed_tokens.weight.numel() * 2)


@require_torch
@require_entropack
@require_torch_gpu
class EntroPackTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls._tmpdir.cleanup)
        cls.tmpdir = cls._tmpdir.name
        cls.base_dir = os.path.join(cls.tmpdir, "base")
        torch.manual_seed(0)
        LlamaForCausalLM(tiny_config()).to(torch.bfloat16).save_pretrained(cls.base_dir)

    def tearDown(self):
        gc.collect()
        backend_empty_cache(torch_device)
        gc.collect()

    @parameterized.expand([("default",), ("fp8",), ("int8",)])
    def test_online_compression_save_reload_and_resave(self, linear_type):
        linear_class = {
            "default": entropack.CompressedLinear,
            "fp8": entropack.CompressedFP8Linear,
            "int8": entropack.CompressedINT8Linear,
        }[linear_type]
        if linear_type != "default" and torch.cuda.get_device_capability() < linear_class.min_capability:
            self.skipTest(f"{linear_type} requires CUDA capability >= {linear_class.min_capability}")
        with patch.object(
            linear_class, "compress_weight", autospec=True, side_effect=linear_class.compress_weight
        ) as compress:
            model = LlamaForCausalLM.from_pretrained(
                self.base_dir,
                quantization_config=EntroPackConfig(target_bpp=4.0, linear_type=linear_type),
                dtype=torch.bfloat16,
                device_map={"": torch_device},
            )
        layer_name = "model.layers.0.self_attn.q_proj"
        layer = model.get_submodule(layer_name)
        weight = next(call.args[1] for call in compress.call_args_list if call.args[0] is layer)
        torch.testing.assert_close(weight.cpu(), disk_tensors(self.base_dir)[f"{layer_name}.weight"], rtol=0, atol=0)
        del compress, layer, weight

        layers = compressed_layers(model)
        self.assertTrue(layers)
        self.assertFalse(any("embed_tokens" in name or "lm_head" in name for name in layers))
        for name in layers:
            module = model.get_submodule(name)
            self.assertIsInstance(module, linear_class)
            self.assertIsInstance(module.weight, entropack.CompressedTensor)
            if linear_type == "default":
                self.assertEqual(module.container_dtype, torch.bfloat16)
                self.assertEqual(module.weight.encoded_dtype, torch.bfloat16)
            else:
                self.assertEqual(module.weight_quant_scale.dtype, torch.float32)
                self.assertTrue(torch.isfinite(module.weight_quant_scale).all())

        input_ids = torch.randint(0, 64, (1, 8), device=torch_device)
        with torch.no_grad():
            logits = model(input_ids).logits
            generated = model.generate(input_ids, max_new_tokens=4, do_sample=False)
        self.assertEqual(logits.shape, (1, 8, 64))
        self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(generated.shape, (1, 12))

        before = {key: value.cpu().clone() for key, value in model.state_dict().items()}
        packed_keys = {f"{name}.weight.entropack_quant_state" for name in layers}
        self.assertTrue(packed_keys.issubset(before))
        for key in packed_keys:
            self.assertEqual(before[key].dtype, torch.uint8)
        self.assertEqual(before["model.embed_tokens.weight"].dtype, torch.bfloat16)
        out_dir = os.path.join(self.tmpdir, f"compressed-{linear_type}")
        model.save_pretrained(out_dir)
        del model
        gc.collect()

        restored = LlamaForCausalLM.from_pretrained(out_dir, dtype=torch.bfloat16, device_map={"": torch_device})
        self.assertEqual(
            restored.config.quantization_config.to_dict(), EntroPackConfig(linear_type=linear_type).to_dict()
        )
        self.assertEqual(compressed_layers(restored), layers)
        with torch.no_grad():
            torch.testing.assert_close(restored(input_ids).logits, logits)
            self.assertTrue(torch.equal(restored.generate(input_ids, max_new_tokens=4, do_sample=False), generated))
        on_disk = disk_tensors(out_dir)
        after = restored.state_dict()
        self.assertEqual(set(before), set(on_disk))
        self.assertEqual(set(before), set(after))
        for key, original in before.items():
            torch.testing.assert_close(on_disk[key], original, rtol=0, atol=0)
            torch.testing.assert_close(after[key].cpu(), original, rtol=0, atol=0)

        resave_dir = os.path.join(self.tmpdir, f"resaved-{linear_type}")
        restored.save_pretrained(resave_dir)
        resaved = disk_tensors(resave_dir)
        self.assertEqual(set(before), set(resaved))
        for key, original in before.items():
            torch.testing.assert_close(resaved[key], original, rtol=0, atol=0)

    @parameterized.expand(
        [
            ("missing_weight", "has no packed weight"),
            ("wrong_shape", "declares shape .* expected"),
            ("missing_scale", "did not load weight_quant_scale"),
        ]
    )
    def test_invalid_checkpoint_is_rejected(self, error, message):
        from safetensors.torch import save_file

        linear_type = "int8" if error == "missing_scale" else "default"
        model = LlamaForCausalLM.from_pretrained(
            self.base_dir,
            quantization_config=EntroPackConfig(linear_type=linear_type),
            dtype=torch.bfloat16,
            device_map={"": torch_device},
        )
        out_dir = os.path.join(self.tmpdir, error)
        model.save_pretrained(out_dir)
        del model
        tensors = disk_tensors(out_dir)
        if error == "missing_scale":
            key = next(key for key in tensors if key.endswith(".weight_quant_scale"))
            tensors.pop(key)
        else:
            key = next(key for key in tensors if key.endswith(".weight.entropack_quant_state"))
            if error == "missing_weight":
                tensors.pop(key)
            else:
                tensors[key] = entropack.compress(
                    torch.randn(32, 64, dtype=torch.bfloat16),
                    entropack.EntroPackLossyConfig(target_bpp=4.0, execution_backend="cuda"),
                ).entropack_quant_state
        save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
        with self.assertRaisesRegex(RuntimeError, message):
            LlamaForCausalLM.from_pretrained(out_dir, dtype=torch.bfloat16, device_map={"": torch_device})

    def test_layers_follow_the_model_dtype(self):
        input_ids = torch.randint(0, 64, (1, 8), device=torch_device)

        def check(model, dtype):
            layers = [model.get_submodule(name) for name in compressed_layers(model)]
            self.assertTrue(layers)
            for layer in layers:
                self.assertEqual(layer.container_dtype, dtype)
                self.assertEqual(layer.weight.encoded_dtype, torch.float16)
            with torch.no_grad():
                logits = model(input_ids).logits
            self.assertEqual(logits.dtype, dtype)
            self.assertTrue(torch.isfinite(logits).all())

        online = LlamaForCausalLM.from_pretrained(
            self.base_dir,
            quantization_config=EntroPackConfig(target_bpp=4.0),
            dtype=torch.float16,
            device_map={"": torch_device},
        )
        check(online, torch.float16)
        out_dir = os.path.join(self.tmpdir, "fp16")
        online.save_pretrained(out_dir)
        del online
        gc.collect()
        check(
            LlamaForCausalLM.from_pretrained(out_dir, dtype=torch.bfloat16, device_map={"": torch_device}),
            torch.bfloat16,
        )

    @parameterized.expand([("auto",), ("cuda",)])
    def test_device_map(self, placement):
        model = LlamaForCausalLM.from_pretrained(
            self.base_dir,
            quantization_config=EntroPackConfig(target_bpp=4.0),
            dtype=torch.bfloat16,
            device_map="auto" if placement == "auto" else {"": torch_device},
        )
        layers = compressed_layers(model)
        self.assertTrue(layers)
        self.assertTrue(
            all(model.get_submodule(name).weight.entropack_quant_state.device.type == "cuda" for name in layers)
        )
        input_ids = torch.randint(0, 64, (1, 8), device=torch_device)
        with torch.no_grad():
            logits = model(input_ids).logits
        self.assertEqual(logits.device.type, "cuda")
        self.assertTrue(torch.isfinite(logits).all())

    def test_modules_to_not_convert_is_honoured(self):
        model = LlamaForCausalLM.from_pretrained(
            self.base_dir,
            quantization_config=EntroPackConfig(target_bpp=4.0, modules_to_not_convert=["gate_proj"]),
            dtype=torch.bfloat16,
            device_map={"": torch_device},
        )
        layers = compressed_layers(model)
        self.assertTrue(layers)
        self.assertFalse(any(name.endswith("gate_proj") for name in layers))
        self.assertTrue(any(name.endswith("up_proj") for name in layers))
        self.assertTrue(any(name.endswith("down_proj") for name in layers))

    @parameterized.expand([("default",), ("fp8",), ("int8",)])
    @require_peft
    def test_lora_optimizer_step(self, linear_type):
        from peft import LoraConfig, get_peft_model

        min_capability = {"fp8": (8, 9), "int8": (8, 0)}.get(linear_type)
        if min_capability is not None and torch.cuda.get_device_capability() < min_capability:
            self.skipTest(f"{linear_type} requires CUDA capability >= {min_capability}")
        model = LlamaForCausalLM.from_pretrained(
            self.base_dir,
            quantization_config=EntroPackConfig(target_bpp=4.0, linear_type=linear_type),
            dtype=torch.bfloat16,
            device_map={"": torch_device},
        )
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"]))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        before = [parameter.detach().clone() for parameter in trainable]
        optimizer = torch.optim.AdamW(trainable, lr=1e-2)
        input_ids = torch.randint(0, 64, (2, 8), device=torch_device)
        loss = model(input_ids=input_ids, labels=input_ids).loss
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in trainable)
        )
        optimizer.step()
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, trainable)))

    @require_peft
    def test_trainer_lora_step(self):
        from peft import LoraConfig, get_peft_model

        from transformers import Trainer, TrainingArguments

        model = LlamaForCausalLM.from_pretrained(
            self.base_dir,
            quantization_config=EntroPackConfig(target_bpp=4.0),
            dtype=torch.bfloat16,
            device_map={"": torch_device},
        )
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"]))
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        before = [parameter.detach().clone() for parameter in trainable]
        dataset = [{"input_ids": tokens, "labels": tokens.clone()} for tokens in torch.randint(0, 64, (4, 8))]
        with tempfile.TemporaryDirectory() as output_dir:
            trainer = Trainer(
                model=model,
                args=TrainingArguments(
                    output_dir=output_dir,
                    max_steps=2,
                    per_device_train_batch_size=2,
                    learning_rate=1e-2,
                    disable_tqdm=True,
                    report_to="none",
                    save_strategy="no",
                ),
                train_dataset=dataset,
            )
            # Keep this test on one GPU instead of entering DataParallel on multi-GPU hosts.
            trainer.args._n_gpu = 1
            trainer.train()
        self.assertEqual(trainer.state.global_step, 2)
        self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, trainable)))


if __name__ == "__main__":
    unittest.main()
