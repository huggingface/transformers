import unittest
import tempfile
import shutil
import numpy as np
import torch
from parameterized import parameterized
from transformers.models.qwen3_asr.processing_qwen3_asr import Qwen3ASRProcessor
from transformers import (
    AutoProcessor,
    AutoTokenizer,
    WhisperFeatureExtractor,
    Qwen2TokenizerFast,
)
from transformers.testing_utils import (
    require_librosa, 
    require_torch, 
    require_torchaudio,
)
from ...test_processing_common import ProcessorTesterMixin

class Qwen3ASRProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = Qwen3ASRProcessor

    @classmethod
    @require_torch
    @require_torchaudio
    def setUpClass(cls):
        cls.checkpoint = "Qwen/Qwen3-ASR-0.6B"
        cls.tmpdirname = tempfile.mkdtemp()
        processor = Qwen3ASRProcessor.from_pretrained(cls.checkpoint) 
        processor.save_pretrained(cls.tmpdirname)

    @require_torch
    @require_torchaudio
    def get_tokenizer(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).tokenizer

    @require_torch
    @require_torchaudio
    def get_feature_extractor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs).feature_extractor

    @require_torch
    @require_torchaudio
    def get_processor(self, **kwargs):
        return AutoProcessor.from_pretrained(self.tmpdirname, **kwargs)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmpdirname)

    @require_torch
    @require_torchaudio
    def test_can_load_various_tokenizers(self):
        processor = Qwen3ASRProcessor.from_pretrained(self.checkpoint)
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        self.assertEqual(processor.tokenizer.__class__, tokenizer.__class__)

    @require_torch
    @require_torchaudio
    def test_save_load_pretrained_default(self):    
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        processor = Qwen3ASRProcessor.from_pretrained(self.checkpoint)
        feature_extractor = processor.feature_extractor

        processor = Qwen3ASRProcessor(tokenizer=tokenizer, feature_extractor=feature_extractor)
        processor.save_pretrained(self.tmpdirname)
        processor = Qwen3ASRProcessor.from_pretrained(self.tmpdirname)

        with tempfile.TemporaryDirectory() as tmpdir:
            processor.save_pretrained(tmpdir)
            reloaded = Qwen3ASRProcessor.from_pretrained(tmpdir)

        self.assertEqual(reloaded.tokenizer.get_vocab(), tokenizer.get_vocab())
        self.assertEqual(reloaded.feature_extractor.to_json_string(), feature_extractor.to_json_string())
        self.assertIsInstance(reloaded.feature_extractor, WhisperFeatureExtractor)
        self.assertIsInstance(reloaded.tokenizer, Qwen2TokenizerFast)

    @require_torch
    @require_torchaudio
    def test_tokenizer_integration(self):
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        prompt = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"
        EXPECTED_OUTPUT = ['This', 'Ġis', 'Ġa', 'Ġtest', 'ĠðŁĺ', 'Ĭ', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ', '9', '2', '0', '0', '0', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.Ċ', 'çĶŁæ´»çļĦ', 'çľŁ', 'è°Ľ', 'æĺ¯', 'Ċ', 'Hi', 'Ġ', 'ĠHello', 'Ċ', 'Hi', 'ĠĠ', 'ĠHello', 'ĊĊ', 'ĠĊĠĠĊ', 'ĠHello', 'Ċ', '<s', '>Ċ', 'hi', '<s', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸Ľ', 'à¸µ', 'ĠĠ', 'Ġ', 'ird', 'ĠĠ', 'Ġ', 'à¸Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']
        tokens = tokenizer.tokenize(prompt)
        self.assertEqual(tokens, EXPECTED_OUTPUT)

    @require_torch
    @require_torchaudio
    def test_chat_template(self):
        processor = AutoProcessor.from_pretrained(self.checkpoint)
        expected_prompt = (
            "<|im_start|>system\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            "<|audio_start|><|audio_pad|><|audio_end|><|im_end|>\n"
            "<|im_start|>assistant\n"
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "audio",
                        "path": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-ASR-Repo/asr_en.wav",
                    },
                ],
            },
        ]
        formatted_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        self.assertEqual(expected_prompt, formatted_prompt)



    ### FOR DEBUGGING ###
    @require_librosa
    def test_apply_chat_template_audio(self):

        processor = self.get_processor()

        batch_messages = [
            [
                {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
                {"role": "user", "content": [{"type": "text", "text": "Describe this."}]},
                {"role": "assistant", "content": [{"type": "text", "text": "It is the sound of"}]},
            ]
        ]

        # this fails because of continue_final_message
        # chat template is correctly loading from model checkpoint: Qwen/Qwen3-ASR-0.6B
        #print(processor.chat_template)
        rendered = processor.apply_chat_template(
            batch_messages,
            continue_final_message=True,    
            tokenize=False,
        )