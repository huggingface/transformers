"""Testing suite for the PyTorch Jais2 model."""

import gc
import unittest

import pytest

from transformers import AutoTokenizer, is_torch_available
from transformers.testing_utils import (
    backend_empty_cache,
    require_flash_attn,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester


if is_torch_available():
    import torch

    from transformers import (
        Jais2Config,
        Jais2ForCausalLM,
        Jais2ForQuestionAnswering,
        Jais2ForSequenceClassification,
        Jais2ForTokenClassification,
        Jais2Model,
    )


class Jais2ModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = Jais2Config
        base_model_class = Jais2Model
        causal_lm_class = Jais2ForCausalLM


@require_torch
class Jais2ModelTest(CausalLMModelTest, unittest.TestCase):
    model_tester_class = Jais2ModelTester
    all_model_classes = (
        (
            Jais2Model,
            Jais2ForCausalLM,
            Jais2ForSequenceClassification,
            Jais2ForTokenClassification,
            Jais2ForQuestionAnswering,
        )
        if is_torch_available()
        else ()
    )

    all_generative_model_classes = (Jais2ForCausalLM,) if is_torch_available() else ()

    pipeline_model_mapping = (
        {
            "feature-extraction": Jais2Model,
            "text-generation": Jais2ForCausalLM,
            "text-classification": Jais2ForSequenceClassification,
            "token-classification": Jais2ForTokenClassification,
            "question-answering": Jais2ForQuestionAnswering,
        }
        if is_torch_available()
        else {}
    )

    # def is_pipeline_test_to_skip(
    #     self,
    #     pipeline_test_case_name,
    #     config_class,
    #     model_architecture,
    #     tokenizer_name,
    #     image_processor_name,
    #     feature_extractor_name,
    #     processor_name,
    # ):
    #     return False


JAIS2_8B_CHECKPOINT = "inceptionai/jais-2-8b"


@require_torch
class Jais2IntegrationTest(unittest.TestCase):
    # Update this path to your local checkpoint
    checkpoint = JAIS2_8B_CHECKPOINT

    def tearDown(self):
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @require_torch_accelerator
    def test_model_logits(self):
        """Test that model outputs expected logits for a known input sequence."""
        model = Jais2ForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        input_text = "The capital of France is"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits.float().cpu()

        # Check shape
        self.assertEqual(logits.shape[0], 1)  # batch size
        self.assertEqual(logits.shape[1], input_ids.shape[1])  # sequence length
        self.assertEqual(logits.shape[2], model.config.vocab_size)  # vocab size

        # Check that logits are not NaN or Inf
        self.assertFalse(torch.isnan(logits).any().item())
        self.assertFalse(torch.isinf(logits).any().item())

        # Print logits stats for debugging (you can record expected values from this)
        print(f"Logits mean: {logits.mean(-1)}")
        print(f"Logits slice [0, -1, :30]: {logits[0, -1, :30]}")

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @require_torch_accelerator
    def test_model_logits_bf16(self):
        """Test model logits in bfloat16 precision."""
        model = Jais2ForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        input_text = "The capital of France is"
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits.float().cpu()

        # Check shape
        self.assertEqual(logits.shape[0], 1)
        self.assertEqual(logits.shape[1], input_ids.shape[1])
        self.assertEqual(logits.shape[2], model.config.vocab_size)

        # Check that logits are not NaN or Inf
        self.assertFalse(torch.isnan(logits).any().item())
        self.assertFalse(torch.isinf(logits).any().item())

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @require_torch_accelerator
    def test_model_generation(self):
        """Test basic text generation with greedy decoding."""
        model = Jais2ForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        prompt = "The capital of France is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # Greedy generation
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Generated text: {generated_text}")

        # Check that generation produced new tokens
        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])

        # Check that the prompt is preserved
        self.assertTrue(generated_text.startswith(prompt))

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @require_torch_accelerator
    def test_model_generation_sdpa(self):
        """Test text generation with SDPA attention implementation."""
        model = Jais2ForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="sdpa",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        prompt = "Artificial intelligence is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"SDPA Generated text: {generated_text}")

        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])
        self.assertTrue(generated_text.startswith(prompt))

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @require_torch_accelerator
    @require_flash_attn
    @pytest.mark.flash_attn_test
    def test_model_generation_flash_attn(self):
        """Test text generation with Flash Attention."""
        model = Jais2ForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        )
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        prompt = "Machine learning models are"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            input_ids,
            max_new_tokens=20,
            do_sample=False,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Flash Attention Generated text: {generated_text}")

        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])
        self.assertTrue(generated_text.startswith(prompt))

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @require_torch_accelerator
    def test_layer_norm(self):
        """Test that LayerNorm is used correctly (Jais2 uses LayerNorm instead of RMSNorm)."""
        model = Jais2ForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
        )

        # Check that the model uses LayerNorm
        self.assertIsInstance(model.model.norm, torch.nn.LayerNorm)

        # Check decoder layers use LayerNorm
        for layer in model.model.layers:
            self.assertIsInstance(layer.input_layernorm, torch.nn.LayerNorm)
            self.assertIsInstance(layer.post_attention_layernorm, torch.nn.LayerNorm)

        # Verify LayerNorm has bias (Jais2 uses bias=True)
        self.assertTrue(model.model.norm.bias is not None)
        self.assertTrue(model.model.layers[0].input_layernorm.bias is not None)

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @require_torch_accelerator
    def test_attention_implementations_consistency(self):
        """Test that different attention implementations produce similar outputs."""
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        prompt = "Hello, how are you?"
        input_ids = tokenizer.encode(prompt, return_tensors="pt")

        # Test with eager attention
        model_eager = Jais2ForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        input_ids_eager = input_ids.to(model_eager.device)

        with torch.no_grad():
            output_eager = model_eager(input_ids_eager).logits.cpu()

        del model_eager
        backend_empty_cache(torch_device)
        gc.collect()

        # Test with SDPA attention
        model_sdpa = Jais2ForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.float32,
            attn_implementation="sdpa",
        )
        input_ids_sdpa = input_ids.to(model_sdpa.device)

        with torch.no_grad():
            output_sdpa = model_sdpa(input_ids_sdpa).logits.cpu()

        del model_sdpa
        backend_empty_cache(torch_device)
        gc.collect()

        # Compare outputs (should be close but not necessarily identical due to numerical differences)
        torch.testing.assert_close(output_eager, output_sdpa, rtol=1e-3, atol=1e-3)

    @slow
    @require_torch_accelerator
    @pytest.mark.torch_compile_test
    def test_compile_static_cache(self):
        """Test torch.compile with static cache."""
        model = Jais2ForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        prompt = "The future of AI is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # Generate with static cache
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
            cache_implementation="static",
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Static cache generated text: {generated_text}")

        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])

        del model
        backend_empty_cache(torch_device)
        gc.collect()

    @slow
    @require_torch_accelerator
    @pytest.mark.torch_export_test
    def test_export_static_cache(self):
        """Test torch.export with static cache."""
        model = Jais2ForCausalLM.from_pretrained(
            self.checkpoint,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)

        prompt = "Deep learning is"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)

        # First verify regular generation works
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=10,
            do_sample=False,
        )

        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"Export test generated text: {generated_text}")

        self.assertGreater(generated_ids.shape[1], input_ids.shape[1])

        del model
        backend_empty_cache(torch_device)
        gc.collect()
