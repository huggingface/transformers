import unittest

from transformers import AutoConfig, AutoTokenizer, is_torch_available
from transformers.file_utils import cached_property
from transformers.models.pegasus.configuration_pegasus import task_specific_params
from transformers.testing_utils import require_sentencepiece, require_tokenizers, require_torch, slow, torch_device
from transformers.utils.logging import ERROR, set_verbosity

from .test_modeling_bart import PGE_ARTICLE
from .test_modeling_common import ModelTesterMixin
from .test_modeling_mbart import AbstractSeq2SeqIntegrationTest


if is_torch_available():
    from transformers import AutoModelForSeq2SeqLM, PegasusConfig, PegasusForConditionalGeneration

XSUM_ENTRY_LONGER = """ The London trio are up for best UK act and best album, as well as getting two nominations in the best song category."We got told like this morning 'Oh I think you're nominated'", said Dappy."And I was like 'Oh yeah, which one?' And now we've got nominated for four awards. I mean, wow!"Bandmate Fazer added: "We thought it's best of us to come down and mingle with everyone and say hello to the cameras. And now we find we've got four nominations."The band have two shots at the best song prize, getting the nod for their Tynchy Stryder collaboration Number One, and single Strong Again.Their album Uncle B will also go up against records by the likes of Beyonce and Kanye West.N-Dubz picked up the best newcomer Mobo in 2007, but female member Tulisa said they wouldn't be too disappointed if they didn't win this time around."At the end of the day we're grateful to be where we are in our careers."If it don't happen then it don't happen - live to fight another day and keep on making albums and hits for the fans."Dappy also revealed they could be performing live several times on the night.The group will be doing Number One and also a possible rendition of the War Child single, I Got Soul.The charity song is a  re-working of The Killers' All These Things That I've Done and is set to feature artists like Chipmunk, Ironik and Pixie Lott.This year's Mobos will be held outside of London for the first time, in Glasgow on 30 September.N-Dubz said they were looking forward to performing for their Scottish fans and boasted about their recent shows north of the border."We just done Edinburgh the other day," said Dappy."We smashed up an N-Dubz show over there. We done Aberdeen about three or four months ago - we smashed up that show over there! Everywhere we go we smash it up!" """

set_verbosity(ERROR)


@require_torch
class ModelTester:
    def __init__(self, parent):
        self.config = PegasusConfig(
            vocab_size=99,
            d_model=24,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=32,
            decoder_ffn_dim=32,
            max_position_embeddings=48,
            add_final_layer_norm=True,
        )

    def prepare_config_and_inputs_for_common(self):
        return self.config, {}


@require_torch
class SelectiveCommonTest(unittest.TestCase):
    all_model_classes = (PegasusForConditionalGeneration,) if is_torch_available() else ()

    test_save_load__keys_to_ignore_on_save = ModelTesterMixin.test_save_load__keys_to_ignore_on_save

    def setUp(self):
        self.model_tester = ModelTester(self)


@require_torch
@require_sentencepiece
@require_tokenizers
class PegasusXSUMIntegrationTest(AbstractSeq2SeqIntegrationTest):
    checkpoint_name = "google/pegasus-xsum"
    src_text = [PGE_ARTICLE, XSUM_ENTRY_LONGER]
    tgt_text = [
        "California's largest electricity provider has turned off power to hundreds of thousands of customers.",
        "Pop group N-Dubz have revealed they were surprised to get four nominations for this year's Mobo Awards.",
    ]

    @cached_property
    def model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint_name).to(torch_device)

    @slow
    def test_pegasus_xsum_summary(self):
        assert self.tokenizer.model_max_length == 512
        inputs = self.tokenizer(self.src_text, return_tensors="pt", truncation=True, max_length=512, padding=True).to(
            torch_device
        )
        assert inputs.input_ids.shape == (2, 421)
        translated_tokens = self.model.generate(**inputs, num_beams=2)
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        assert self.tgt_text == decoded

        if "cuda" not in torch_device:
            return
        # Demonstrate fp16 issue, Contributions welcome!
        self.model.half()
        translated_tokens_fp16 = self.model.generate(**inputs, max_length=10)
        decoded_fp16 = self.tokenizer.batch_decode(translated_tokens_fp16, skip_special_tokens=True)
        assert decoded_fp16 == [
            "California's largest electricity provider has begun",
            "N-Dubz have revealed they were",
        ]


class PegasusConfigTests(unittest.TestCase):
    @slow
    def test_task_specific_params(self):
        """Test that task_specific params['summarization_xsum'] == config['pegasus_xsum'] """
        failures = []
        pegasus_prefix = "google/pegasus"
        n_prefix_chars = len("summarization_")
        for task, desired_settings in task_specific_params.items():
            dataset = task[n_prefix_chars:]
            mname = f"{pegasus_prefix}-{dataset}"
            cfg = AutoConfig.from_pretrained(mname)
            for k, v in desired_settings.items():
                actual_value = getattr(cfg, k)
                if actual_value != v:
                    failures.append(f"config for {mname} had {k}: {actual_value}, expected {v}")
            tokenizer = AutoTokenizer.from_pretrained(mname)
            n_pos_embeds = desired_settings["max_position_embeddings"]
            if n_pos_embeds != tokenizer.model_max_length:
                failures.append(f"tokenizer.model_max_length {tokenizer.model_max_length} expected {n_pos_embeds}")

        # error
        all_fails = "\n".join(failures)
        assert not failures, f"The following configs have unexpected settings: {all_fails}"
