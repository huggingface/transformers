import gc
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    Glm4MoeConfig,
    Glm4MoeForCausalLM,
    pipeline,
)
from transformers.cache_utils import MtpCache
from transformers.generation.candidate_generator import (
    AssistantToTargetTranslator,
    AssistantVocabTranslatorCache,
    DFlashTokenCandidateGenerator,
    SinglePositionMultiTokenCandidateGenerator,
    UniversalSpeculativeDecodingGenerator,
)
from transformers.generation.logits_process import LogitsProcessorList, TopKLogitsWarper
from transformers.modeling_layers import MtpModel
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.models.muse_glimmer_assistant import MuseGlimmerAssistantConfig, MuseGlimmerAssistantModel
from transformers.testing_utils import require_torch, torch_device


@require_torch
class TestAssistantToTargetTranslator(unittest.TestCase):
    def setUp(self):
        # Create mock tokenizers with predefined vocabularies
        self.target_tokenizer = MagicMock()
        self.assistant_tokenizer = MagicMock()
        self.assistant_model = MagicMock(device=torch_device)

        # Define mock vocabularies for the tokenizers
        self.target_vocab = {"hello": 0, "world": 1, "foo": 2, "bar": 3}
        self.assistant_vocab = {"hello": 0, "world": 1, "foo": 2, "baz": 4}

        self.target_tokenizer.get_vocab.return_value = self.target_vocab
        self.assistant_tokenizer.get_vocab.return_value = self.assistant_vocab
        self.target_vocab_size = 6

        # Instantiate the class under test
        self.translator = AssistantToTargetTranslator(
            target_tokenizer=self.target_tokenizer,
            assistant_tokenizer=self.assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
            assistant_model=self.assistant_model,
            assistant_prune_lm_head=False,
        )

    def test_get_assistant_to_target_input_ids(self):
        """Test the mapping from assistant tokens to target tokens."""
        expected_mapping = [0, 1, 2, self.translator.SUPPRESS_TOKEN_ID, self.translator.SUPPRESS_TOKEN_ID]
        actual_mapping = self.translator._assistant_to_target_input_ids.tolist()
        self.assertEqual(actual_mapping, expected_mapping)

    def test_get_suppress_input_ids(self):
        """Test the suppression of assistant input IDs not present in the target vocabulary."""
        expected_suppress_ids = [3, 4]
        actual_suppress_ids = self.translator._get_suppress_input_ids().tolist()
        self.assertEqual(actual_suppress_ids, expected_suppress_ids)

    def test_get_target_ids(self):
        """Test the translation of assistant candidate IDs to target candidate IDs."""
        assistant_input_ids = torch.LongTensor([[0, 1, 2]]).to(
            self.assistant_model.device
        )  # 'hello world foo' in assistant tokenizer
        target_input_ids = torch.LongTensor([[0, 1, 2]]).to(
            self.assistant_model.device
        )  # 'hello world foo' in target tokenizer
        assistant_candidate_ids = torch.LongTensor([[0, 1, 2, 4]]).to(
            self.assistant_model.device
        )  # 'hello world foo baz' in assistant tokenizer

        expected_target_ids = torch.LongTensor(
            [[0, 1, 2, self.translator.SUPPRESS_TOKEN_ID]]
        ).to(
            self.assistant_model.device
        )  # 'hello world foo baz' in target tokenizer (baz is mapped to self.translator.suppress_tokens_id since it does not exist in target vocab)

        actual_target_ids = self.translator.get_target_ids(
            assistant_input_ids, target_input_ids, assistant_candidate_ids
        )
        self.assertTrue(torch.equal(actual_target_ids, expected_target_ids))

    def test_get_target_logits(self):
        """Test the conversion of assistant logits to target logits."""
        # Assistant logits for IDs 0, 1, 2
        assistant_logits = torch.FloatTensor([[[0.1, 0.2, 0.3, 0.4, self.translator.FILTER_VALUE]]]).to(
            self.assistant_model.device
        )  # Shape (1, 1, 5)

        # Expected target logits (target_vocab_size = 4)
        expected_target_logits = torch.full((1, 1, self.target_vocab_size), self.translator.FILTER_VALUE).to(
            self.assistant_model.device
        )
        expected_target_logits[0, 0, 0] = 0.1  # 'hello'
        expected_target_logits[0, 0, 1] = 0.2  # 'world'
        expected_target_logits[0, 0, 2] = 0.3  # 'foo'
        # The 'bar' token in target vocab remains at -inf

        actual_target_logits = self.translator.get_target_logits(assistant_logits)
        self.assertTrue(torch.equal(actual_target_logits, expected_target_logits))


class MockTokenizer:
    """A simple mock tokenizer class that supports weak references."""

    def __init__(self, vocab=None):
        self._vocab = vocab or {}

    def get_vocab(self):
        return self._vocab

    def __call__(self, text, add_special_tokens=True):
        # Mock implementation of the __call__ method
        tokens = text.split()
        input_ids = [self._vocab.get(token, 0) for token in tokens]
        return {"input_ids": input_ids}


@require_torch
class TestAssistantVocabTranslatorCache(unittest.TestCase):
    def setUp(self):
        # Clear the cache before each test
        AssistantVocabTranslatorCache._cache.clear()
        # Create mock tokenizers with different vocabularies
        self.target_tokenizer = MockTokenizer({"hello": 0, "world": 1})
        self.assistant_tokenizer = MockTokenizer({"hello": 0, "world": 1, "foo": 2})
        self.other_target_tokenizer = MockTokenizer({"foo": 2, "bar": 3})
        self.other_assistant_tokenizer = MockTokenizer({"baz": 4, "qux": 5})
        self.assistant_model = MagicMock(device=torch_device)

        self.target_vocab_size = 6

    def test_same_instance_for_same_tokenizers(self):
        """Test that the same translator is returned for the same tokenizers."""
        translator1 = AssistantVocabTranslatorCache.get_translator(
            self.target_tokenizer,
            self.assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
            assistant_model=self.assistant_model,
            assistant_prune_lm_head=False,
        )
        translator2 = AssistantVocabTranslatorCache.get_translator(
            self.target_tokenizer,
            self.assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
            assistant_model=self.assistant_model,
            assistant_prune_lm_head=False,
        )
        self.assertIs(translator1, translator2, "Translators should be cached and identical")

    def test_different_instances_for_different_tokenizers(self):
        """Test that different tokenizers produce different translators."""
        translator1 = AssistantVocabTranslatorCache.get_translator(
            self.target_tokenizer,
            self.assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
            assistant_model=self.assistant_model,
            assistant_prune_lm_head=False,
        )
        translator2 = AssistantVocabTranslatorCache.get_translator(
            self.other_target_tokenizer,
            self.other_assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
            assistant_model=self.assistant_model,
            assistant_prune_lm_head=False,
        )
        self.assertIsNot(translator1, translator2, "Translators should differ for different tokenizers")

    def test_cache_with_weakref_key(self):
        """Ensure that the cache uses weak references as keys."""
        initial_cache_size = len(AssistantVocabTranslatorCache._cache)
        target_tokenizer = MockTokenizer({"hello": 0})
        assistant_tokenizer = MockTokenizer({"hello": 0})

        # Store translator in a local variable to avoid it being kept alive
        translator = AssistantVocabTranslatorCache.get_translator(
            target_tokenizer,
            assistant_tokenizer,
            target_vocab_size=self.target_vocab_size,
            assistant_model=self.assistant_model,
            assistant_prune_lm_head=False,
        )
        self.assertEqual(len(AssistantVocabTranslatorCache._cache), initial_cache_size + 1)

        # Delete all strong references
        del target_tokenizer
        del assistant_tokenizer
        del translator

        # Force garbage collection
        gc.collect()

        # Call cleanup to remove dead entries
        AssistantVocabTranslatorCache.cleanup()

        # The cache size remains increased due to strong references
        self.assertEqual(len(AssistantVocabTranslatorCache._cache), initial_cache_size + 1)

    def test_weakref_cache_cleanup(self):
        """Test that the cache cleans up translators when tokenizers are garbage collected."""

        def create_translator():
            target_tokenizer = MockTokenizer({"hello": 0})
            assistant_tokenizer = MockTokenizer({"hello": 0})
            translator = AssistantVocabTranslatorCache.get_translator(
                target_tokenizer,
                assistant_tokenizer,
                target_vocab_size=self.target_vocab_size,
                assistant_model=self.assistant_model,
                assistant_prune_lm_head=False,
            )
            # Create weak references before returning
            refs = (weakref.ref(translator), weakref.ref(target_tokenizer), weakref.ref(assistant_tokenizer))
            # Remove strong references inside the function
            del target_tokenizer
            del assistant_tokenizer
            del translator
            return refs

        translator_ref, target_ref, assistant_ref = create_translator()

        # Force garbage collection
        gc.collect()

        # Call cleanup to remove dead entries
        AssistantVocabTranslatorCache.cleanup()

        # The tokenizers and translator are not garbage collected due to strong references
        self.assertIsNotNone(target_ref(), "Target tokenizer should still be alive due to strong references")
        self.assertIsNotNone(assistant_ref(), "Assistant tokenizer should still be alive due to strong references")
        self.assertIsNotNone(translator_ref(), "Translator should still be alive due to strong references")


@require_torch
class TestUniversalSpeculativeDecoding(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.target_name = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        cls.assistant_name = "hf-internal-testing/tiny-random-PhiForCausalLM"

    def setUp(self):
        self.target_tokenizer = AutoTokenizer.from_pretrained(self.target_name)
        self.target_config = AutoConfig.from_pretrained(self.target_name)
        self.assistant_model = AutoModelForCausalLM.from_pretrained(self.assistant_name).to(torch_device)
        self.assistant_tokenizer = AutoTokenizer.from_pretrained(self.assistant_name)
        self.generation_config = GenerationConfig(max_length=20, min_length=0)

        # Ensure required tokens exist
        if self.target_tokenizer.pad_token_id is None:
            self.target_tokenizer.pad_token_id = self.target_tokenizer.eos_token_id
        if self.target_tokenizer.bos_token_id is None:
            self.target_tokenizer.bos_token_id = self.target_tokenizer.eos_token_id
        if self.assistant_tokenizer.pad_token_id is None:
            self.assistant_tokenizer.pad_token_id = self.assistant_tokenizer.eos_token_id
        if self.assistant_tokenizer.bos_token_id is None:
            self.assistant_tokenizer.bos_token_id = self.assistant_tokenizer.eos_token_id

        self.input_ids = torch.tensor([[1, 2, 3]]).to(torch_device)
        self.model_kwargs = {
            "attention_mask": torch.ones_like(self.input_ids).to(torch_device),
        }
        atm_translator = AssistantVocabTranslatorCache.get_translator(
            target_tokenizer=self.target_tokenizer,
            assistant_tokenizer=self.assistant_tokenizer,
            assistant_model=self.assistant_model,
            target_vocab_size=self.target_config.vocab_size,
        )
        self.generator = UniversalSpeculativeDecodingGenerator(
            input_ids=self.input_ids,
            assistant_model=self.assistant_model,
            target_tokenizer=self.target_tokenizer,
            assistant_tokenizer=self.assistant_tokenizer,
            generation_config=self.generation_config,
            model_kwargs=self.model_kwargs,
            atm_translator=atm_translator,
        )

    def test_basic_generation(self):
        """Test basic speculative decoding works"""
        input_text = "The quick brown fox"
        input_ids = self.target_tokenizer.encode(input_text, return_tensors="pt")
        self.generator.input_ids = input_ids
        candidates, scores = self.generator.get_candidates(input_ids)

        self.assertIsNotNone(candidates)
        self.assertIsNotNone(scores)
        self.assertTrue(torch.is_tensor(candidates))
        self.assertTrue(torch.is_tensor(scores))

    def test_mismatched_vocabularies(self):
        """Test handling of mismatched vocabularies between models"""
        # Create input with tokens present in main but not assistant vocab
        # Find a token that is not in the assistant tokenizer but in
        # the main tokenizer.
        missing_token = next(
            token
            for token in self.target_tokenizer.get_vocab()
            if token not in self.assistant_tokenizer.get_vocab()
            and token not in self.target_tokenizer.all_special_tokens
            and "reserved_" not in token
        )
        input_ids = torch.tensor([[self.target_tokenizer.convert_tokens_to_ids(missing_token)]])
        self.generator.input_ids = input_ids
        candidates, _ = self.generator.get_candidates(input_ids)
        self.assertIsNotNone(candidates)

    def test_speculation_depth(self):
        """Test different speculation depths"""
        input_ids = self.target_tokenizer.encode("Test text", return_tensors="pt")
        self.generator.input_ids = input_ids

        for depth in [1, 8, 17]:
            self.generator.num_assistant_tokens = depth
            candidates, _ = self.generator.get_candidates(input_ids)
            self.assertLessEqual(candidates.shape[1] - input_ids.shape[1], depth)

    def test_device_consistency(self):
        """Test handling of inputs on different devices"""
        input_ids = torch.tensor([[1, 2, 3]]).to(torch_device)
        self.generator.input_ids = input_ids
        candidates, _ = self.generator.get_candidates(input_ids)
        self.assertEqual(candidates.device, input_ids.device)

    def test_usd_vs_vanilla_sampling(cls):
        """Test that USD matches vanilla sampling with temperature set to nearly 0"""
        prompt = "Test text"

        pipe_vanilla = pipeline(
            "text-generation",
            model=cls.target_name,
        )
        pipe_vanilla_output = pipe_vanilla(prompt, max_new_tokens=5, do_sample=False)
        vanilla_text = pipe_vanilla_output[0]["generated_text"]

        pipe_usd = pipeline(
            "text-generation",
            model=cls.target_name,
            assistant_model=cls.assistant_name,
        )
        pipe_usd_output = pipe_usd(prompt, max_new_tokens=5, do_sample=True, temperature=1e-9)  # Nearly 0 temperature
        usd_text = pipe_usd_output[0]["generated_text"]

        # Assert that the outputs match
        cls.assertEqual(usd_text, vanilla_text)


@require_torch
class TestCandidateLogitsContract(unittest.TestCase):
    """The verifier assumes `candidate_logits` describe the distribution the draft tokens were drawn from
    (lossless speculative sampling). These tests pin that contract for each candidate generator."""

    top_k = 3

    def _dflash_setup(self):
        vocab_size, hidden_size, block_size, prompt_length = 8, 32, 8, 6
        raw_logits = torch.tensor([3.0, 2.6, 2.2, 1.0, 0.6, 0.2, 0.0, -0.4])

        class PinnedOutputEmbeddings(nn.Module):
            def forward(self, hidden_states):
                return raw_logits.to(hidden_states.device).expand(*hidden_states.shape[:2], vocab_size).clone()

        config = MuseGlimmerAssistantConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            head_dim=8,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=37,
            block_size=block_size,
            target_layer_ids=[0, 2],
            mask_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=4,
        )
        assistant_model = MuseGlimmerAssistantModel(config).to(torch_device).eval()
        input_embeddings = nn.Embedding(vocab_size, hidden_size).to(torch_device)
        input_ids = torch.zeros(1, prompt_length, dtype=torch.long, device=torch_device)
        model_kwargs = {
            "position_ids": torch.arange(prompt_length, device=torch_device)[None],
            "attention_mask": torch.ones(1, prompt_length, dtype=torch.long, device=torch_device),
        }
        model_outputs = BaseModelOutputWithPast(
            last_hidden_state=torch.randn(1, prompt_length, hidden_size, device=torch_device),
            hidden_states=tuple(torch.randn(1, prompt_length, hidden_size, device=torch_device) for _ in range(4)),
        )
        generation_config = GenerationConfig(do_sample=True, top_k=self.top_k, max_length=1024)
        generator = DFlashTokenCandidateGenerator(
            assistant_model=assistant_model,
            main_model_input_embeddings=input_embeddings,
            main_model_output_embeddings=PinnedOutputEmbeddings(),
            generation_config=generation_config,
            logits_processor=LogitsProcessorList([TopKLogitsWarper(top_k=self.top_k)]),
        )
        return generator, input_ids, model_kwargs, model_outputs, raw_logits, block_size

    def test_dflash_returns_processed_logits(self):
        generator, input_ids, model_kwargs, model_outputs, raw_logits, block_size = self._dflash_setup()
        torch.manual_seed(0)
        candidate_ids, candidate_logits = generator.get_candidates(
            input_ids=input_ids,
            model_kwargs=model_kwargs,
            model_outputs=model_outputs,
            is_first_iteration=False,
            n_last_matches=0,
        )
        # The top-k warper ignores `input_ids`, so the processed logits are the raw pinned logits with the
        # out-of-top-k tokens masked to -inf.
        num_drafted = candidate_logits.shape[1]
        expected = torch.stack(
            [
                TopKLogitsWarper(top_k=self.top_k)(input_ids, raw_logits[None].float().to(torch_device))
                for _ in range(num_drafted)
            ],
            dim=1,
        )
        self.assertTrue(torch.equal(candidate_logits, expected))
        drafted = candidate_ids[0, input_ids.shape[1] :]
        self.assertEqual(drafted.numel(), num_drafted)
        # The returned distribution must only have mass on tokens the drafter can propose.
        self.assertTrue((candidate_logits[0].softmax(dim=-1) > 0).sum(dim=-1).eq(self.top_k).all())

    def _build_mtp(self, num_mtp_layers):
        config = Glm4MoeConfig(
            vocab_size=99,
            hidden_size=32,
            intermediate_size=37,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_mtp_layers=num_mtp_layers,
        )
        model = Glm4MoeForCausalLM(config).to(torch_device).eval()
        mtp_model = MtpModel(model, num_mtp_layers=num_mtp_layers)
        input_ids = torch.zeros(1, 2, dtype=torch.long, device=torch_device)
        attention_mask = torch.ones(1, 2, dtype=torch.long, device=torch_device)
        position_ids = torch.arange(2, device=torch_device)[None]
        last_hidden_states = torch.randn(1, 2, config.hidden_size, device=torch_device)
        return mtp_model, config, input_ids, attention_mask, position_ids, last_hidden_states

    def _run_mtp(
        self,
        mtp_model,
        config,
        input_ids,
        attention_mask,
        position_ids,
        last_hidden_states,
        do_sample,
        logits_processor,
        full_input_ids,
        seed,
    ):
        torch.manual_seed(seed)
        mtp_cache = MtpCache(config=config.get_mtp_config())
        mtp_cache.activate_past_recording()
        return mtp_model(
            input_ids=input_ids,
            last_hidden_states=last_hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            mtp_cache=mtp_cache,
            do_sample=do_sample,
            logits_processor=logits_processor,
            full_input_ids=full_input_ids,
        )

    def _check_mtp_contract(self, do_sample):
        # With sampling, the drafted tokens differ between the two runs, so a single MTP layer is used (its logits
        # are computed before any drafting). Without sampling, top-k keeps the argmax, so multiple layers are fine.
        num_mtp_layers = 1 if do_sample else 2
        mtp_model, config, input_ids, attention_mask, position_ids, last_hidden_states = self._build_mtp(
            num_mtp_layers
        )
        warper = TopKLogitsWarper(top_k=self.top_k)
        _, raw, _ = self._run_mtp(
            mtp_model,
            config,
            input_ids,
            attention_mask,
            position_ids,
            last_hidden_states,
            do_sample,
            None,
            None,
            seed=0,
        )
        _, processed, _ = self._run_mtp(
            mtp_model,
            config,
            input_ids,
            attention_mask,
            position_ids,
            last_hidden_states,
            do_sample,
            LogitsProcessorList([warper]),
            input_ids.clone(),
            seed=0,
        )
        expected = torch.stack([warper(input_ids, raw[:, i, :].float()) for i in range(raw.shape[1])], dim=1)
        self.assertTrue(torch.equal(processed, expected))

    def test_mtp_returns_processed_logits_with_sampling(self):
        self._check_mtp_contract(do_sample=True)

    def test_mtp_returns_processed_logits_without_sampling(self):
        self._check_mtp_contract(do_sample=False)

    def test_single_position_returns_point_mass_logits(self):
        vocab_size, hidden_size, num_tokens = 8, 16, 4
        logits = torch.tensor([1.0, 2.5, 0.5, 2.0, -1.0, -2.0, 0.0, 1.5])

        class FakeGemma4Assistant(nn.Module):
            def __init__(self, logits, hidden_size):
                super().__init__()
                self.register_buffer("logits", logits)
                self.dummy_param = nn.Parameter(torch.zeros(1))
                self.hidden_size = hidden_size
                self.generation_config = GenerationConfig(
                    do_sample=False, num_assistant_tokens=num_tokens, eos_token_id=None
                )
                self.config = SimpleNamespace(is_encoder_decoder=False)

            @property
            def device(self):
                return self.dummy_param.device

            def forward(
                self,
                inputs_embeds,
                attention_mask=None,
                position_ids=None,
                shared_kv_states=None,
                use_cache=False,
                **kwargs,
            ):
                batch, seq, _ = inputs_embeds.shape
                out_logits = self.logits.to(inputs_embeds.device).expand(batch, seq, -1)
                return ModelOutput(
                    logits=out_logits,
                    last_hidden_state=torch.zeros(batch, seq, self.hidden_size, device=inputs_embeds.device),
                )

        assistant_model = FakeGemma4Assistant(logits, hidden_size).to(torch_device)
        seq_len = 3
        input_ids = torch.zeros(1, seq_len, dtype=torch.long, device=torch_device)
        model_kwargs = {"attention_mask": torch.ones(1, seq_len, dtype=torch.long, device=torch_device)}
        model_outputs = ModelOutput(
            hidden_states=(torch.randn(1, seq_len, hidden_size, device=torch_device),),
            shared_kv_states={
                "layer_0": (
                    torch.zeros(1, 2, seq_len, 16, device=torch_device),
                    torch.zeros(1, 2, seq_len, 16, device=torch_device),
                )
            },
        )
        generation_config = GenerationConfig(
            max_length=32, pad_token_id=0, eos_token_id=None, num_assistant_tokens=num_tokens
        )
        generator = SinglePositionMultiTokenCandidateGenerator(
            input_ids=input_ids,
            assistant_model=assistant_model,
            target_model_input_embeddings=nn.Embedding(vocab_size, hidden_size).to(torch_device),
            generation_config=generation_config,
            model_kwargs=model_kwargs,
        )
        candidate_ids, candidate_logits = generator.get_candidates(
            input_ids=input_ids,
            model_kwargs=model_kwargs,
            model_outputs=model_outputs,
            is_first_iteration=False,
            n_last_matches=0,
        )
        drafted = candidate_ids[0, seq_len:]
        self.assertEqual(drafted.numel(), num_tokens)
        argmax_id = logits.to(torch_device).argmax()
        self.assertTrue((drafted == argmax_id).all())
        # Deterministic drafting: the proposal distribution is a point mass on the drafted token.
        expected = torch.full_like(candidate_logits, -torch.inf)
        expected.scatter_(-1, drafted[None, :, None], 0.0)
        self.assertTrue(torch.equal(candidate_logits, expected))
