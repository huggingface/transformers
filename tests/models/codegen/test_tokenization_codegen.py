import tempfile
import unittest

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers import AutoTokenizer
from transformers.models.codegen.tokenization_codegen import CodeGenTokenizer
from transformers.testing_utils import (
    require_tokenizers,
)


# Master input string of combined test cases
input_string = """This is a test
I was born in 92000, and this is falsé.
生活的真谛是
Hi  Hello
Hi   Hello

 
  
 Hello
<s>
hi<s>there
The following string should be properly encoded: Hello.
But ird and ปี   ird   ด
Hey how are you doing"""


# Expected tokens from expected_tokens_codegen.py
expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ92', '000', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶŁ', 'æ', '´', '»', 'çļĦ', 'çľ', 'Ł', 'è', '°', 'Ľ', 'æĺ¯', 'Ċ', 'Hi', '  ', 'Hello', 'Ċ', 'Hi', '   ', 'Hello', 'ĊĊ', 'Ġ', 'Ċ', '  ', 'Ċ', 'ĠHello', 'Ċ', '<', 's', '>', 'Ċ', 'hi', '<', 's', '>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠHello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', '   ', 'ird', '   ', 'à¸', 'Ķ', 'Ċ', 'Hey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']
expected_token_ids = [1212, 318, 257, 1332, 198, 40, 373, 4642, 287, 10190, 830, 11, 290, 428, 318, 27807, 2634, 13, 198, 37955, 162, 112, 119, 21410, 40367, 253, 164, 108, 249, 42468, 198, 17250, 50286, 15496, 198, 17250, 50285, 15496, 628, 220, 198, 50286, 198, 18435, 198, 27, 82, 29, 198, 5303, 27, 82, 29, 8117, 198, 464, 1708, 4731, 815, 307, 6105, 30240, 25, 18435, 13, 198, 1537, 220, 1447, 290, 220, 19567, 249, 19567, 113, 50285, 1447, 50285, 19567, 242, 198, 10814, 703, 389, 345, 1804]


@require_tokenizers
class CodeGenTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    # TokenizerTesterMixin configuration
    from_pretrained_id = ["Salesforce/codegen-350M-mono"]
    tokenizer_class = CodeGenTokenizer
    rust_tokenizer_class = CodeGenTokenizer
    test_rust_tokenizer = False
    from_pretrained_kwargs = {}

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        from_pretrained_id = "Salesforce/codegen-350M-mono"

        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tok_auto.pad_token = tok_auto.eos_token
        tok_auto.save_pretrained(cls.tmpdirname)

        cls.tokenizers = [tok_auto]

    def get_tokenizers(self, **kwargs):
        kwargs.setdefault("pad_token", "<PAD>")
        return super().get_tokenizers(**kwargs)

    def test_integration_expected_tokens(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.tokenize(input_string), expected_tokens)

    def test_integration_expected_token_ids(self):
        for tok in self.tokenizers:
            self.assertEqual(tok.encode(input_string), expected_token_ids)

    def test_save_and_reload(self):
        for tok in self.tokenizers:
            with self.subTest(f"{tok.__class__.__name__}"):
                original_tokens = tok.tokenize(input_string)
                original_ids = tok.encode(input_string)
                
                # Save tokenizer to temporary directory
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tok.save_pretrained(tmp_dir)
                    
                    # Reload tokenizer from saved directory
                    reloaded_tok = tok.__class__.from_pretrained(tmp_dir)
                    
                    # Test that reloaded tokenizer produces same results
                    reloaded_tokens = reloaded_tok.tokenize(input_string)
                    reloaded_ids = reloaded_tok.encode(input_string)
                    
                    self.assertEqual(original_tokens, reloaded_tokens)
                    self.assertEqual(original_ids, reloaded_ids)

    def test_backend_parameter(self):
        """Test that the backend parameter works correctly with AutoTokenizer."""
        from_pretrained_id = "Salesforce/codegen-350M-mono"
        
        test_text = "Hello world! This is a test."
        
        tok_auto = AutoTokenizer.from_pretrained(from_pretrained_id)
        tokens_auto = tok_auto.tokenize(test_text)
        
        tok_tokenizers = AutoTokenizer.from_pretrained(from_pretrained_id, backend="tokenizers")
        tokens_tokenizers = tok_tokenizers.tokenize(test_text)
        
        self.assertEqual(tokens_auto, tokens_tokenizers)
        
        self.assertEqual(tok_auto.__class__.__name__, "CodeGenTokenizer")
        self.assertEqual(tok_tokenizers.__class__.__name__, "CodeGenTokenizer")
