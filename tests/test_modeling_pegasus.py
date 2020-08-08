import unittest

from transformers import AutoConfig
from transformers.testing_utils import require_torch, slow, torch_device

from .test_modeling_bart import PGE_ARTICLE
from .test_modeling_mbart import AbstractSeq2SeqIntegrationTest


@require_torch
class PegasusIntegrationTest(AbstractSeq2SeqIntegrationTest):
    checkpoint_name = "sshleifer/pegasus/xsum"
    src_text = PGE_ARTICLE
    tgt_text = "California's largest electricity provider has turned off power to tens of thousands of customers in an effort to reduce the risk of wildfires."

    @slow
    def test_pegasus_xsum_summary(self):
        inputs = self.tokenizer([self.src_text], return_tensors="pt").to(torch_device)
        translated_tokens = self.model.generate(input_ids=inputs["input_ids"].to(torch_device),)
        decoded = self.tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        self.assertEqual(self.tgt_text, decoded[0])


class PegasusConfigTests(unittest.TestCase):
    def test_all_config_max_lengths(self):
        expected_max_length = {
            # See appendix C of paper
            "xsum": 64,
            "cnn_dailymail": 128,
            "newsroom": 128,
            "wikihow": 256,
            "multinews": 256,
            "reddit_tifu": 128,
            "big_patent": 256,
            "arxiv": 256,
            "pubmed": 256,
            "gigaword": 32,
            "aeslc": 32,
            "billsum": 256,
        }
        failures = []
        pegasus_prefix = "sshleifer/pegasus"
        for dataset, max_len in expected_max_length.items():
            mname = f"{pegasus_prefix}/{dataset}"
            cfg = AutoConfig.from_pretrained(mname)
            if cfg.max_length != max_len:
                failures.append(f"config for {mname} had max_length: {cfg.max_length}, expected {max_len}")
        if failures == []:
            return
        # error
        all_fails = "\n".join(failures)
        raise AssertionError(f"The following configs have unexpected settings: {all_fails}")
