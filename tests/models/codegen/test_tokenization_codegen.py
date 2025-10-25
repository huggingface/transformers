import tempfile
import unittest

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers import AutoTokenizer
from transformers.models.codegen.tokenization_codegen import CodeGenTokenizer
from transformers.testing_utils import (
    require_tokenizers,
)



@require_tokenizers
class CodeGenTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    # TokenizerTesterMixin configuration
    from_pretrained_id = ["Salesforce/codegen-350M-mono"]
    tokenizer_class = CodeGenTokenizer
    from_pretrained_kwargs = {}


    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', '  ', 'Hello', 'Ċ', 'Hi', '   ', 'Hello', 'ĊĊ', 'Ġ', 'Ċ', '  ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', '   ', 'ird', '   ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']
    integration_expected_token_ids = [1212, 318, 257, 1332, 198, 40, 373, 4642, 287, 10190, 830, 11, 290, 428, 318, 27807, 2634, 13, 198, 37955, 162, 112, 119, 21410, 40367, 253, 164, 108, 249, 42468, 198, 17250, 50286, 15496, 198, 17250, 50285, 15496, 628, 220, 198, 50286, 198, 18435, 198, 27, 82, 29, 198, 5303, 27, 82, 29, 8117, 198, 464, 1708, 4731, 815, 307, 6105, 30240, 25, 18435, 13, 198, 1537, 220, 1447, 290, 220, 19567, 249, 19567, 113, 50285, 1447, 50285, 19567, 242, 198, 10814, 703, 389, 345, 1804]
    integration_expected_decoded_text = 'This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing'
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "Salesforce/codegen-350M-mono"

        tokenizer = CodeGenTokenizer.from_pretrained(from_pretrained_id)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tokenizer]

    def get_tokenizers(self, **kwargs):
        kwargs.setdefault("pad_token", "<PAD>")
        return super().get_tokenizers(**kwargs)

    def test_backend_parameter(self):
        """Test that the backend parameter works correctly with AutoTokenizer."""
        from_pretrained_id = "Salesforce/codegen-350M-mono"
        
        test_text = "Hello world! This is a test."
        
        tokenizer = CodeGenTokenizer.from_pretrained(from_pretrained_id)
        tokens_auto = tokenizer.tokenize(test_text)
        
        tok_tokenizers = CodeGenTokenizer.from_pretrained(from_pretrained_id, backend="tokenizers")
        tokens_tokenizers = tok_tokenizers.tokenize(test_text)
        
        self.assertEqual(tokens_auto, tokens_tokenizers)
        
        self.assertEqual(tokenizer.__class__.__name__, "CodeGenTokenizer")
        self.assertEqual(tok_tokenizers.__class__.__name__, "CodeGenTokenizer")
