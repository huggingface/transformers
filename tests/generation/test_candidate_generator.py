import gc
import unittest
import weakref
from unittest.mock import MagicMock, patch

import torch
from torch import nn

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DeepseekV3Config,
    DeepseekV3ForCausalLM,
    GenerationConfig,
    MuseGlimmerAssistantConfig,
    MuseGlimmerAssistantModel,
    pipeline,
)
from transformers.generation.candidate_generator import (
    AssistantToTargetTranslator,
    AssistantVocabTranslatorCache,
    DFlashTokenCandidateGenerator,
    MTPCandidateGenerator,
    UniversalSpeculativeDecodingGenerator,
)
from transformers.generation.logits_process import LogitsProcessorList
from transformers.modeling_layers import MtpModel
from transformers.modeling_outputs import BaseModelOutputWithPast
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
class TestDFlashTokenCandidateGenerator(unittest.TestCase):
    """`_assisted_decoding` commits a whole block before checking whether to stop, and that check only looks at
    the last committed token. A block holding an EOS anywhere else would therefore be accepted whole and
    generation would run past it, so the drafter has to crop its block at the first EOS.
    """

    vocab_size = 50
    hidden_size = 32
    block_size = 8
    prompt_length = 6
    # `block_size - 1` drafted tokens, with a token that never repeats sitting in the middle to stand in as EOS
    draft = [11, 12, 13, 7, 14, 15, 16]

    def setUp(self):
        torch.manual_seed(0)
        config = MuseGlimmerAssistantConfig(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            head_dim=8,
            num_hidden_layers=3,
            num_attention_heads=4,
            num_key_value_heads=2,
            intermediate_size=37,
            block_size=self.block_size,
            target_layer_ids=[0, 2],
            mask_token_id=1,
            bos_token_id=2,
            eos_token_id=3,
            pad_token_id=4,
        )
        self.assistant_model = MuseGlimmerAssistantModel(config).to(torch_device).eval()
        self.input_embeddings = nn.Embedding(self.vocab_size, self.hidden_size).to(torch_device)
        self.input_ids = torch.randint(5, self.vocab_size, (1, self.prompt_length), device=torch_device)
        self.model_kwargs = {
            "position_ids": torch.arange(self.prompt_length, device=torch_device)[None],
            "attention_mask": torch.ones(1, self.prompt_length, dtype=torch.long, device=torch_device),
        }
        # `target_layer_ids` indexes into these with an offset of one, so layer 2 needs four entries
        self.model_outputs = BaseModelOutputWithPast(
            last_hidden_state=torch.randn(1, self.prompt_length, self.hidden_size, device=torch_device),
            hidden_states=tuple(
                torch.randn(1, self.prompt_length, self.hidden_size, device=torch_device) for _ in range(4)
            ),
        )

    def _get_candidates(self, eos_token_id, logits_processor=None, draft=None):
        """Drafts one block, pinning the vocab projection so the drafted tokens are exactly `draft`."""

        draft = self.draft if draft is None else draft
        vocab_size = self.vocab_size

        class _PinnedOutputEmbeddings(nn.Module):
            def __init__(self):
                super().__init__()
                # `get_candidates` reads `.weight.device` to place hidden states (#47877)
                self.weight = nn.Parameter(torch.zeros(1, device=torch_device))

            def forward(self, hidden_states):
                logits = torch.zeros(*hidden_states.shape[:2], vocab_size, device=hidden_states.device)
                for position, token in enumerate(draft):
                    # The anchor position is dropped before the head since #48007, so rows map 1:1 to the draft
                    logits[:, position, token] = 1.0
                return logits

        generation_config = GenerationConfig(do_sample=False, max_length=1024)
        generation_config._eos_token_tensor = (
            None if eos_token_id is None else torch.tensor([eos_token_id], device=torch_device)
        )
        candidate_generator = DFlashTokenCandidateGenerator(
            assistant_model=self.assistant_model,
            main_model_input_embeddings=self.input_embeddings,
            main_model_output_embeddings=_PinnedOutputEmbeddings(),
            generation_config=generation_config,
            logits_processor=logits_processor,
        )
        candidate_ids, candidate_logits = candidate_generator.get_candidates(
            input_ids=self.input_ids,
            model_kwargs=self.model_kwargs,
            model_outputs=self.model_outputs,
            is_first_iteration=False,
            n_last_matches=0,
        )
        return candidate_ids[0, self.prompt_length :].tolist(), candidate_logits

    def _assert_logits_track_tokens(self, logits, drafted):
        """The logits are consumed positionally alongside the tokens, so they must be cropped the same way.

        Checking the length alone would not notice logits cropped from the wrong end, which
        `_speculative_sampling` would then read per-token against the wrong distribution.
        """
        self.assertEqual(logits.shape[1], len(drafted))
        self.assertEqual(logits.argmax(-1)[0].tolist(), drafted)

    def test_draft_is_cropped_at_the_first_eos(self):
        for logits_processor in (None, LogitsProcessorList()):
            with self.subTest(logits_processor=logits_processor):
                # `self.draft` holds the EOS at index 3, so four tokens survive
                drafted, logits = self._get_candidates(eos_token_id=7, logits_processor=logits_processor)
                self.assertEqual(drafted, [11, 12, 13, 7])
                self._assert_logits_track_tokens(logits, drafted)

    def test_draft_is_cropped_at_the_first_of_several_eos(self):
        # A block drafter routinely repeats a token, so "first" has to be pinned by a draft that can
        # actually tell it apart from "last" -- `self.draft` has distinct entries and cannot.
        drafted, logits = self._get_candidates(eos_token_id=7, draft=[11, 12, 7, 13, 7, 14, 15])
        self.assertEqual(drafted, [11, 12, 7])
        self._assert_logits_track_tokens(logits, drafted)

    def test_draft_without_an_eos_is_untouched(self):
        for eos_token_id in (None, 42):
            with self.subTest(eos_token_id=eos_token_id):
                drafted, logits = self._get_candidates(eos_token_id=eos_token_id)
                self.assertEqual(drafted, self.draft)
                self._assert_logits_track_tokens(logits, drafted)

    def test_draft_already_ending_in_eos_is_untouched(self):
        drafted, logits = self._get_candidates(eos_token_id=self.draft[-1])
        self.assertEqual(drafted, self.draft)
        self._assert_logits_track_tokens(logits, drafted)

    def test_draft_is_cropped_to_a_leading_eos(self):
        drafted, logits = self._get_candidates(eos_token_id=self.draft[0])
        self.assertEqual(drafted, self.draft[:1])
        self._assert_logits_track_tokens(logits, drafted)


@require_torch
class TestMTPCandidateGenerator(unittest.TestCase):
    """`_assisted_decoding` commits `n_matches + 1` tokens before checking whether to stop, and that check
    reads only the last of them, so an EOS earlier in the drafted block is committed along with everything
    after it. Reachable only with more than one mtp layer: with a single layer the drafted EOS is already
    last, and the `is_done_candidate` guard in `_assisted_decoding` covers it.
    """

    vocab_size = 50
    hidden_size = 32
    prompt_length = 6
    draft = [11, 12, 7, 13, 14]  # one drafted token per mtp layer

    def setUp(self):
        torch.manual_seed(0)
        config = DeepseekV3Config(
            vocab_size=self.vocab_size, hidden_size=self.hidden_size, intermediate_size=37,
            moe_intermediate_size=12, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
            n_routed_experts=4, n_shared_experts=1, num_experts_per_tok=2, first_k_dense_replace=1,
            n_group=1, topk_group=1, q_lora_rank=8, kv_lora_rank=8, qk_rope_head_dim=4, v_head_dim=8,
            qk_nope_head_dim=4, num_mtp_layers=len(self.draft), pad_token_id=2,
        )  # fmt: skip
        self.model = DeepseekV3ForCausalLM(config).to(torch_device).eval()
        self.input_ids = torch.randint(5, self.vocab_size, (1, self.prompt_length), device=torch_device)
        self.model_kwargs = {
            "position_ids": torch.arange(self.prompt_length, device=torch_device)[None],
            "attention_mask": torch.ones(1, self.prompt_length, dtype=torch.long, device=torch_device),
        }
        self.model_outputs = BaseModelOutputWithPast(
            last_hidden_state=torch.randn(1, self.prompt_length, self.hidden_size, device=torch_device),
            hidden_states=tuple(
                torch.randn(1, self.prompt_length, self.hidden_size, device=torch_device) for _ in range(3)
            ),
        )

    def _get_drafted_tokens(self, eos_token_id, draft=None):
        """Drafts one block, pinning the shared vocab projection so the drafted tokens are exactly `draft`."""
        draft = self.draft if draft is None else draft
        vocab_size = self.vocab_size

        class _PinnedHead(nn.Module):
            """Each mtp layer asks the head, which the drafter shares with the main model, for one token."""

            layer = 0

            def forward(self, hidden_states):
                logits = torch.zeros(*hidden_states.shape[:2], vocab_size, device=hidden_states.device)
                logits[:, -1, draft[self.layer]] = 1.0
                self.layer += 1
                return logits

        self.model.lm_head = _PinnedHead()
        generation_config = GenerationConfig(do_sample=False, max_length=1024)
        generation_config._eos_token_tensor = (
            None if eos_token_id is None else torch.tensor([eos_token_id], device=torch_device)
        )
        # The drafter is normally read out of the checkpoint; its weights do not matter here, as the pinned
        # head is what decides the drafted tokens
        with patch.object(
            MtpModel, "from_pretrained", lambda model, **kwargs: MtpModel(model, len(draft)).to(torch_device)
        ):
            candidate_generator = MTPCandidateGenerator(
                main_model=self.model,
                generation_config=generation_config,
                model_kwargs=self.model_kwargs,
                logits_processor=LogitsProcessorList(),
            )
        candidate_ids, candidate_logits = candidate_generator.get_candidates(
            input_ids=self.input_ids,
            model_kwargs=self.model_kwargs,
            model_outputs=self.model_outputs,
            is_first_iteration=False,
            n_last_matches=0,
        )
        drafted = candidate_ids[0, self.prompt_length :].tolist()
        # `_speculative_sampling` reads the logits per token, so they have to be cropped alongside them --
        # matching lengths alone would not catch logits cropped from the wrong end
        self.assertEqual(candidate_logits.argmax(-1)[0].tolist(), drafted)
        return drafted

    def test_draft_is_cropped_at_the_first_eos(self):
        for eos_token_id, draft, expected in (
            (7, None, [11, 12, 7]),
            # a drafter repeats tokens, and only a repeated EOS tells "first" apart from "last"
            (7, [11, 7, 12, 7, 13], [11, 7]),
            (11, None, [11]),
        ):
            with self.subTest(eos_token_id=eos_token_id, draft=draft):
                self.assertEqual(self._get_drafted_tokens(eos_token_id, draft), expected)

    def test_draft_without_an_eos_before_the_end_is_untouched(self):
        for eos_token_id in (None, 42, 14):  # unset, not drafted, already the last drafted token
            with self.subTest(eos_token_id=eos_token_id):
                self.assertEqual(self._get_drafted_tokens(eos_token_id), self.draft)
