# Copyright 2025 the HuggingFace Team. All rights reserved.
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
"""Testing suite for the PyTorch DeepSeekV3.2 model."""

import unittest

import pytest
from parameterized import parameterized

from transformers import Cache, is_torch_available
from transformers.testing_utils import require_torch, require_torch_accelerator, slow

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_modeling_common import (
    TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION,
    TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION,
)


if is_torch_available():
    import torch

    from transformers import (
        AutoTokenizer,
        DeepseekV32ForCausalLM,
        DeepseekV32Model,
    )


# Opening of "Alice's Adventures in Wonderland" by Lewis Carroll (public domain, Project Gutenberg).
# Used as a real, coherent long-context prompt (~2.8K tokens > index_topk) for the DSA indexer test.
LONG_CONTEXT_PROMPT = """Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do: once or twice she had peeped into the book her sister was reading, but it had no pictures or conversations in it, “and what is the use of a book,” thought Alice “without pictures or conversations?”

So she was considering in her own mind (as well as she could, for the hot day made her feel very sleepy and stupid), whether the pleasure of making a daisy-chain would be worth the trouble of getting up and picking the daisies, when suddenly a White Rabbit with pink eyes ran close by her.

There was nothing so _very_ remarkable in that; nor did Alice think it so _very_ much out of the way to hear the Rabbit say to itself, “Oh dear! Oh dear! I shall be late!” (when she thought it over afterwards, it occurred to her that she ought to have wondered at this, but at the time it all seemed quite natural); but when the Rabbit actually _took a watch out of its waistcoat-pocket_, and looked at it, and then hurried on, Alice started to her feet, for it flashed across her mind that she had never before seen a rabbit with either a waistcoat-pocket, or a watch to take out of it, and burning with curiosity, she ran across the field after it, and fortunately was just in time to see it pop down a large rabbit-hole under the hedge.

In another moment down went Alice after it, never once considering how in the world she was to get out again.

The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well.

Either the well was very deep, or she fell very slowly, for she had plenty of time as she went down to look about her and to wonder what was going to happen next. First, she tried to look down and make out what she was coming to, but it was too dark to see anything; then she looked at the sides of the well, and noticed that they were filled with cupboards and book-shelves; here and there she saw maps and pictures hung upon pegs. She took down a jar from one of the shelves as she passed; it was labelled “ORANGE MARMALADE”, but to her great disappointment it was empty: she did not like to drop the jar for fear of killing somebody underneath, so managed to put it into one of the cupboards as she fell past it.

“Well!” thought Alice to herself, “after such a fall as this, I shall think nothing of tumbling down stairs! How brave they’ll all think me at home! Why, I wouldn’t say anything about it, even if I fell off the top of the house!” (Which was very likely true.)

Down, down, down. Would the fall _never_ come to an end? “I wonder how many miles I’ve fallen by this time?” she said aloud. “I must be getting somewhere near the centre of the earth. Let me see: that would be four thousand miles down, I think—” (for, you see, Alice had learnt several things of this sort in her lessons in the schoolroom, and though this was not a _very_ good opportunity for showing off her knowledge, as there was no one to listen to her, still it was good practice to say it over) “—yes, that’s about the right distance—but then I wonder what Latitude or Longitude I’ve got to?” (Alice had no idea what Latitude was, or Longitude either, but thought they were nice grand words to say.)

Presently she began again. “I wonder if I shall fall right _through_ the earth! How funny it’ll seem to come out among the people that walk with their heads downward! The Antipathies, I think—” (she was rather glad there _was_ no one listening, this time, as it didn’t sound at all the right word) “—but I shall have to ask them what the name of the country is, you know. Please, Ma’am, is this New Zealand or Australia?” (and she tried to curtsey as she spoke—fancy _curtseying_ as you’re falling through the air! Do you think you could manage it?) “And what an ignorant little girl she’ll think me for asking! No, it’ll never do to ask: perhaps I shall see it written up somewhere.”

Down, down, down. There was nothing else to do, so Alice soon began talking again. “Dinah’ll miss me very much to-night, I should think!” (Dinah was the cat.) “I hope they’ll remember her saucer of milk at tea-time. Dinah my dear! I wish you were down here with me! There are no mice in the air, I’m afraid, but you might catch a bat, and that’s very like a mouse, you know. But do cats eat bats, I wonder?” And here Alice began to get rather sleepy, and went on saying to herself, in a dreamy sort of way, “Do cats eat bats? Do cats eat bats?” and sometimes, “Do bats eat cats?” for, you see, as she couldn’t answer either question, it didn’t much matter which way she put it. She felt that she was dozing off, and had just begun to dream that she was walking hand in hand with Dinah, and saying to her very earnestly, “Now, Dinah, tell me the truth: did you ever eat a bat?” when suddenly, thump! thump! down she came upon a heap of sticks and dry leaves, and the fall was over.

Alice was not a bit hurt, and she jumped up on to her feet in a moment: she looked up, but it was all dark overhead; before her was another long passage, and the White Rabbit was still in sight, hurrying down it. There was not a moment to be lost: away went Alice like the wind, and was just in time to hear it say, as it turned a corner, “Oh my ears and whiskers, how late it’s getting!” She was close behind it when she turned the corner, but the Rabbit was no longer to be seen: she found herself in a long, low hall, which was lit up by a row of lamps hanging from the roof.

There were doors all round the hall, but they were all locked; and when Alice had been all the way down one side and up the other, trying every door, she walked sadly down the middle, wondering how she was ever to get out again.

Suddenly she came upon a little three-legged table, all made of solid glass; there was nothing on it except a tiny golden key, and Alice’s first thought was that it might belong to one of the doors of the hall; but, alas! either the locks were too large, or the key was too small, but at any rate it would not open any of them. However, on the second time round, she came upon a low curtain she had not noticed before, and behind it was a little door about fifteen inches high: she tried the little golden key in the lock, and to her great delight it fitted!

Alice opened the door and found that it led into a small passage, not much larger than a rat-hole: she knelt down and looked along the passage into the loveliest garden you ever saw. How she longed to get out of that dark hall, and wander about among those beds of bright flowers and those cool fountains, but she could not even get her head through the doorway; “and even if my head would go through,” thought poor Alice, “it would be of very little use without my shoulders. Oh, how I wish I could shut up like a telescope! I think I could, if I only knew how to begin.” For, you see, so many out-of-the-way things had happened lately, that Alice had begun to think that very few things indeed were really impossible.

There seemed to be no use in waiting by the little door, so she went back to the table, half hoping she might find another key on it, or at any rate a book of rules for shutting people up like telescopes: this time she found a little bottle on it, (“which certainly was not here before,” said Alice,) and round the neck of the bottle was a paper label, with the words “DRINK ME,” beautifully printed on it in large letters.

It was all very well to say “Drink me,” but the wise little Alice was not going to do _that_ in a hurry. “No, I’ll look first,” she said, “and see whether it’s marked ‘_poison_’ or not”; for she had read several nice little histories about children who had got burnt, and eaten up by wild beasts and other unpleasant things, all because they _would_ not remember the simple rules their friends had taught them: such as, that a red-hot poker will burn you if you hold it too long; and that if you cut your finger _very_ deeply with a knife, it usually bleeds; and she had never forgotten that, if you drink much from a bottle marked “poison,” it is almost certain to disagree with you, sooner or later.

However, this bottle was _not_ marked “poison,” so Alice ventured to taste it, and finding it very nice, (it had, in fact, a sort of mixed flavour of cherry-tart, custard, pine-apple, roast turkey, toffee, and hot buttered toast,) she very soon finished it off.

* * * * * * *

* * * * * *

* * * * * * *

“What a curious feeling!” said Alice; “I must be shutting up like a telescope.”

And so it was indeed: she was now only ten inches high, and her face brightened up at the thought that she was now the right size for going through the little door into that lovely garden. First, however, she waited for a few minutes to see if she was going to shrink any further: she felt a little nervous about this; “for it might end, you know,” said Alice to herself, “in my going out altogether, like a candle. I wonder what I should be like then?” And she tried to fancy what the flame of a candle is like after the candle is blown out, for she could not remember ever having seen such a thing.

After a while, finding that nothing more happened, she decided on going into the garden at once; but, alas for poor Alice! when she got to the door, she found she had forgotten the little golden key, and when she went back to the table for it, she found she could not possibly reach it: she could see it quite plainly through the glass, and she tried her best to climb up one of the legs of the table, but it was too slippery; and when she had tired herself out with trying, the poor little thing sat down and cried.

“Come, there’s no use in crying like that!” said Alice to herself, rather sharply; “I advise you to leave off this minute!” She generally gave herself very good advice, (though she very seldom followed it), and sometimes she scolded herself so severely as to bring tears into her eyes; and once she remembered trying to box her own ears for having cheated herself in a game of croquet she was playing against herself, for this curious child was very fond of pretending to be two people. “But it’s no use now,” thought poor Alice, “to pretend to be two people! Why, there’s hardly enough of me left to make _one_ respectable person!”

Soon her eye fell on a little glass box that was lying under the table: she opened it, and found in it a very small cake, on which the words “EAT ME” were beautifully marked in currants. “Well, I’ll eat it,” said Alice, “and if it makes me grow larger, I can reach the key; and if it makes me grow smaller, I can creep under the door; so either way I’ll get into the garden, and I don’t care which happens!”

She ate a little bit, and said anxiously to herself, “Which way? Which way?”, holding her hand on the top of her head to feel which way it was growing, and she was quite surprised to find that she remained the same size: to be sure, this generally happens when one eats cake, but Alice had got so much into the way of expecting nothing but out-of-the-way things to happen, that it seemed quite dull and stupid for life to go on in the common way.

So she set to work, and very soon finished off the cake.

* * * * * * *

* * * * * *

* * * * * * *



CHAPTER II. The Pool of Tears

“Curiouser and curiouser!” cried Alice (she was so much surprised, that for the moment she quite forgot how to speak good English); “now I’m opening out like the largest telescope that ever was! Good-bye, feet!” (for when she looked down at her feet, they seemed to be almost out of sight, they were getting so far off)."""


class DeepseekV32ModelTester(CausalLMModelTester):
    if is_torch_available():
        base_model_class = DeepseekV32Model

    def __init__(
        self,
        parent,
        n_routed_experts=8,
        kv_lora_rank=32,
        q_lora_rank=16,
        qk_nope_head_dim=64,
        qk_rope_head_dim=64,
        first_k_dense_replace=1,
        n_group=1,
        topk_group=1,
    ):
        super().__init__(parent=parent)
        self.n_routed_experts = n_routed_experts
        self.kv_lora_rank = kv_lora_rank
        self.q_lora_rank = q_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.first_k_dense_replace = first_k_dense_replace
        self.n_group = n_group
        self.topk_group = topk_group


@require_torch
class DeepseekV32ModelTest(CausalLMModelTest, unittest.TestCase):
    pipeline_model_mapping = (
        {
            "feature-extraction": DeepseekV32Model,
            "text-generation": DeepseekV32ForCausalLM,
        }
        if is_torch_available()
        else {}
    )
    fx_compatible = False
    test_torchscript = False
    test_all_params_have_gradient = False
    model_tester_class = DeepseekV32ModelTester
    model_split_percents = [0.5, 0.7, 0.8]

    # used in `test_torch_compile_for_training`
    _torch_compile_train_cls = DeepseekV32ForCausalLM if is_torch_available() else None

    @unittest.skip("DeepseekV32 applies RoPE to qk_rope_head_dim; generic rope scaling tests assume config.head_dim")
    def test_model_rope_scaling_frequencies(self):
        pass

    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    @unittest.skip("DeepseekV32 applies RoPE to qk_rope_head_dim; generic rope scaling tests assume config.head_dim")
    def test_model_rope_scaling_from_config(self, scaling_type):
        pass

    def _check_past_key_values_for_generate(self, batch_size, past_key_values, seq_length, config):
        """Needs to be overridden as deepseek has special MLA cache format (though we don't really use the MLA)"""
        self.assertIsInstance(past_key_values, Cache)

        # (batch, head, seq_length, head_features)
        expected_common_shape = (
            batch_size,
            getattr(config, "num_key_value_heads", config.num_attention_heads),
            seq_length,
        )
        expected_key_shape = expected_common_shape + (config.qk_nope_head_dim + config.qk_rope_head_dim,)
        expected_value_shape = expected_common_shape + (config.v_head_dim,)

        for layer in past_key_values.layers:
            self.assertEqual(layer.keys.shape, expected_key_shape)
            self.assertEqual(layer.values.shape, expected_value_shape)

    @parameterized.expand([("random",), ("same",)])
    @unittest.skip("DeepseekV32 uses MLA so it is not compatible with assisted decoding")
    def test_assisted_decoding_matches_greedy_search(self, assistant_type):
        pass

    @unittest.skip("DeepseekV32 uses MLA so it is not compatible with assisted decoding")
    def test_prompt_lookup_decoding_matches_greedy_search(self):
        pass

    @unittest.skip("DeepseekV32 uses MLA so it is not compatible with assisted decoding")
    def test_assisted_decoding_sample(self):
        pass

    @unittest.skip("DeepseekV32 uses MLA so it is not compatible with the standard cache format")
    def test_beam_search_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip("DeepseekV32 uses MLA so it is not compatible with the standard cache format")
    def test_greedy_generate_dict_outputs_use_cache(self):
        pass

    @unittest.skip(reason="SDPA can't dispatch on flash due to unsupported head dims")
    def test_sdpa_can_dispatch_on_flash(self):
        pass

    @unittest.skip("Dynamic control flow in MoE")
    @pytest.mark.torch_compile_test
    def test_torch_compile_for_training(self):
        pass

    # DeepSeek Sparse Attention selects tokens with a hard top-k, which is discontinuous: a tiny numerical
    # difference in the indexer scores (attention backend, padding, batching, sequence packing) can flip
    # which tokens are selected and thus change the output. These exact cross-backend / padding-equivalence
    # tests therefore do not hold for DSA (dense models like DeepSeek-V3 pass them).
    @parameterized.expand(TEST_EAGER_MATCHES_SDPA_INFERENCE_PARAMETERIZATION)
    @unittest.skip("DSA hard top-k selection is sensitive to tiny numerical differences across backends.")
    def test_eager_matches_sdpa_inference(self, *args, **kwargs):
        pass

    @parameterized.expand(TEST_EAGER_MATCHES_BATCHED_AND_GROUPED_INFERENCE_PARAMETERIZATION)
    @unittest.skip("DSA hard top-k selection is sensitive to tiny numerical differences across batching.")
    def test_eager_matches_batched_and_grouped_inference(self, *args, **kwargs):
        pass

    @unittest.skip("DSA hard top-k selection is sensitive to padding shifts (selection can flip).")
    def test_left_padding_compatibility(self):
        pass

    @unittest.skip("DSA hard top-k selection is sensitive to sequence packing (selection can flip).")
    def test_eager_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("DSA hard top-k selection is sensitive to sequence packing (selection can flip).")
    def test_sdpa_padding_matches_padding_free_with_position_ids(self):
        pass

    @unittest.skip("MoE routing on a tiny randomly-initialized model makes the overfit target unstable.")
    def test_training_overfit(self):
        pass


@slow
@require_torch_accelerator
class DeepseekV32IntegrationTest(unittest.TestCase):
    def test_deepseek_v32(self):
        EXPECTED_TEXT = ['An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors. The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.\n\nWe call our particular attention "Scaled Dot-Product Attention" (Figure (left']  # fmt: skip

        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")
        model = DeepseekV32ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V3.2-Exp",
            device_map="auto",
            dtype=torch.bfloat16,
        )

        input_text = [
            "An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors."  # fmt: skip
        ]
        model_inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

        generated_ids = model.generate(**model_inputs, max_new_tokens=50, do_sample=False)
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        self.assertEqual(generated_text, EXPECTED_TEXT)

    def test_logits_eager(self):
        input_ids = [1, 306, 4658, 278, 6593, 310, 2834, 338]

        model = DeepseekV32ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V3.2-Exp",
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        with torch.no_grad():
            out = model(torch.tensor([input_ids]).to(model.device))

        EXPECTED_MEAN = torch.tensor([[5.0182, 6.1787, 5.3601, 5.7569, 5.7146, 5.1751, 4.2580, 3.3002]], device=out.logits.device)  # fmt: skip
        torch.testing.assert_close(out.logits.float().mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3)

        EXPECTED_SLICE = torch.tensor([17.3750, 12.4375,  2.1406, 15.0625, 13.5625, 14.8750, 13.7500, 13.6250, 13.5625, 14.0000, 13.0000, 15.1875, 13.6250, 13.3750, 15.3750], device=out.logits.device)  # fmt: skip
        torch.testing.assert_close(out.logits[0, 0, :15].float(), EXPECTED_SLICE, atol=1e-3, rtol=1e-3)

    def test_logits_long_context(self):
        prompt = LONG_CONTEXT_PROMPT

        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3.2-Exp")
        model = DeepseekV32ForCausalLM.from_pretrained(
            "deepseek-ai/DeepSeek-V3.2-Exp",
            device_map="auto",
            dtype=torch.bfloat16,
            attn_implementation="eager",
        )

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        self.assertGreater(inputs["input_ids"].shape[1], model.config.index_topk)

        with torch.no_grad():
            out = model(**inputs)

        EXPECTED_MEAN = torch.tensor([1.6388, 0.5092, 1.8607, 1.3185, 0.3267, 1.0360, 2.3001, 3.5533], device=out.logits.device)  # fmt: skip
        torch.testing.assert_close(out.logits[0, -8:].float().mean(-1), EXPECTED_MEAN, atol=1e-3, rtol=1e-3)

        EXPECTED_SLICE = torch.tensor([-2.8594, 23.7500, -0.8086, 18.1250, 19.8750, 15.8125, 12.8750, 14.6875, 21.3750, 19.0000, 20.3750, 20.0000, 18.7500, 16.3750, 22.0000], device=out.logits.device)  # fmt: skip
        torch.testing.assert_close(out.logits[0, -1, :15].float(), EXPECTED_SLICE, atol=1e-3, rtol=1e-3)

        gen = model.generate(**inputs, max_new_tokens=40, do_sample=False)
        continuation = tokenizer.decode(gen[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        EXPECTED_GENERATION = " “Oh, my poor little feet, I wonder who will put on your shoes and stockings for you now, dears? I’m sure _I_ shan’t be able!"  # fmt: skip
        self.assertEqual(EXPECTED_GENERATION, continuation)
