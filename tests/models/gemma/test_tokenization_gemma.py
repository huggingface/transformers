# Copyright 2024 The HuggingFace Team. All rights reserved.
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


import unittest

from tests.test_tokenization_common import TokenizerTesterMixin
from transformers.models.gemma.tokenization_gemma import GemmaTokenizer
from transformers.testing_utils import (
    require_read_token,
    require_tokenizers,
)


@require_tokenizers
@require_read_token
class GemmaTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "google/gemma-7b"
    tokenizer_class = GemmaTokenizer

    integration_expected_tokens = [
        "This",
        "▁is",
        "▁a",
        "▁test",
        "▁😊",
        "\n",
        "I",
        "▁was",
        "▁born",
        "▁in",
        "▁",
        "9",
        "2",
        "0",
        "0",
        "0",
        ",",
        "▁and",
        "▁this",
        "▁is",
        "▁fals",
        "é",
        ".",
        "\n",
        "生活的",
        "真",
        "谛",
        "是",
        "\n",
        "Hi",
        "▁▁",
        "Hello",
        "\n",
        "Hi",
        "▁▁▁",
        "Hello",
        "\n\n",
        "▁",
        "\n",
        "▁▁",
        "\n",
        "▁Hello",
        "\n",
        "<s>",
        "\n",
        "hi",
        "<s>",
        "there",
        "\n",
        "The",
        "▁following",
        "▁string",
        "▁should",
        "▁be",
        "▁properly",
        "▁encoded",
        ":",
        "▁Hello",
        ".",
        "\n",
        "But",
        "▁i",
        "rd",
        "▁and",
        "▁ปี",
        "▁▁▁",
        "ird",
        "▁▁▁",
        "ด",
        "\n",
        "Hey",
        "▁how",
        "▁are",
        "▁you",
        "▁doing",
    ]
    integration_expected_token_ids = [
        1596,
        603,
        476,
        2121,
        44416,
        108,
        235285,
        729,
        7565,
        575,
        235248,
        235315,
        235284,
        235276,
        235276,
        235276,
        235269,
        578,
        736,
        603,
        40751,
        235335,
        235265,
        108,
        122182,
        235710,
        245467,
        235427,
        108,
        2151,
        139,
        4521,
        108,
        2151,
        140,
        4521,
        109,
        235248,
        108,
        139,
        108,
        25957,
        108,
        204,
        108,
        544,
        204,
        11048,
        108,
        651,
        2412,
        2067,
        1412,
        614,
        10338,
        49748,
        235292,
        25957,
        235265,
        108,
        1860,
        496,
        1924,
        578,
        73208,
        140,
        5650,
        140,
        235732,
        108,
        6750,
        1368,
        708,
        692,
        3900,
    ]
    expected_tokens_from_ids = [
        "This",
        "▁is",
        "▁a",
        "▁test",
        "▁😊",
        "\n",
        "I",
        "▁was",
        "▁born",
        "▁in",
        "▁",
        "9",
        "2",
        "0",
        "0",
        "0",
        ",",
        "▁and",
        "▁this",
        "▁is",
        "▁fals",
        "é",
        ".",
        "\n",
        "生活的",
        "真",
        "谛",
        "是",
        "\n",
        "Hi",
        "▁▁",
        "Hello",
        "\n",
        "Hi",
        "▁▁▁",
        "Hello",
        "\n\n",
        "▁",
        "\n",
        "▁▁",
        "\n",
        "▁Hello",
        "\n",
        "<s>",
        "\n",
        "hi",
        "<s>",
        "there",
        "\n",
        "The",
        "▁following",
        "▁string",
        "▁should",
        "▁be",
        "▁properly",
        "▁encoded",
        ":",
        "▁Hello",
        ".",
        "\n",
        "But",
        "▁i",
        "rd",
        "▁and",
        "▁ปี",
        "▁▁▁",
        "ird",
        "▁▁▁",
        "ด",
        "\n",
        "Hey",
        "▁how",
        "▁are",
        "▁you",
        "▁doing",
    ]
    integration_expected_decoded_text = "This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing"

    def test_internal_consistency(self):
        """Override to add debug output on failure."""
        import os
        
        tokenizer = self.get_tokenizer()
        
        def get_debug_info():
            """Build debug info string to include in error messages."""
            debug_lines = []
            debug_lines.append(f"Tokenizer type: {type(tokenizer).__name__}")
            debug_lines.append(f"Tokenizer module: {type(tokenizer).__module__}")
            debug_lines.append(f"Tokenizer class location: {tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}")
            
            if hasattr(tokenizer, 'name_or_path'):
                debug_lines.append(f"Tokenizer name_or_path: {tokenizer.name_or_path}")
                # Check what files exist in the temp directory
                if tokenizer.name_or_path and os.path.exists(tokenizer.name_or_path):
                    debug_lines.append(f"Temp directory exists: True")
                    try:
                        files = os.listdir(tokenizer.name_or_path)
                        debug_lines.append(f"Files in temp directory: {', '.join(files)}")
                        tokenizer_json_path = os.path.join(tokenizer.name_or_path, "tokenizer.json")
                        if os.path.exists(tokenizer_json_path):
                            debug_lines.append(f"tokenizer.json exists: True")
                            debug_lines.append(f"tokenizer.json size: {os.path.getsize(tokenizer_json_path)} bytes")
                            # Try to read and check vocab size from the file
                            try:
                                import json
                                with open(tokenizer_json_path, 'r') as f:
                                    tj = json.load(f)
                                if 'model' in tj and 'vocab' in tj['model']:
                                    vocab_size = len(tj['model']['vocab'])
                                    debug_lines.append(f"Vocab size in tokenizer.json: {vocab_size}")
                            except Exception as e:
                                debug_lines.append(f"Error reading tokenizer.json: {e}")
                        else:
                            debug_lines.append(f"tokenizer.json exists: False")
                    except Exception as e:
                        debug_lines.append(f"Error listing temp directory: {e}")
                else:
                    debug_lines.append(f"Temp directory exists: False")
            
            if hasattr(tokenizer, 'vocab_file'):
                debug_lines.append(f"Tokenizer vocab_file: {tokenizer.vocab_file}")
            
            if hasattr(tokenizer, '_tokenizer'):
                debug_lines.append(f"Has _tokenizer attribute: True")
                debug_lines.append(f"_tokenizer type: {type(tokenizer._tokenizer)}")
                if hasattr(tokenizer._tokenizer, 'model'):
                    debug_lines.append(f"_tokenizer.model type: {type(tokenizer._tokenizer.model)}")
                    # Try to get vocab size from the model
                    try:
                        if hasattr(tokenizer._tokenizer.model, 'get_vocab'):
                            vocab_dict = tokenizer._tokenizer.model.get_vocab()
                            debug_lines.append(f"_tokenizer.model vocab size: {len(vocab_dict)}")
                    except:
                        pass
                # Try to get more details about the tokenizer
                try:
                    debug_lines.append(f"_tokenizer object: {repr(tokenizer._tokenizer)}")
                except:
                    debug_lines.append(f"_tokenizer object: <repr failed>")
            else:
                debug_lines.append(f"Has _tokenizer attribute: False")
            
            # Add debug info from tokenization_utils_base if available
            if hasattr(tokenizer, '_gemma_debug_info'):
                debug_lines.append("\nGEMMA_DEBUG_INFO from tokenization_utils_base:")
                for key, value in tokenizer._gemma_debug_info.items():
                    debug_lines.append(f"  {key}: {value}")
            
            debug_lines.append(f"Vocab size: {len(tokenizer)}")
            return "\n".join(debug_lines)
        
        # Now run the actual test
        try:
            input_text, output_text = self.get_input_output_texts(tokenizer)

            tokens = tokenizer.tokenize(input_text)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            ids_2 = tokenizer.encode(input_text, add_special_tokens=False)
            try:
                self.assertListEqual(ids, ids_2)
            except AssertionError as e:
                debug_info = get_debug_info()
                raise AssertionError(f"{e}\n\nDEBUG INFO:\n{debug_info}")

            tokens_2 = tokenizer.convert_ids_to_tokens(ids)
            try:
                self.assertNotEqual(len(tokens_2), 0)
            except AssertionError as e:
                debug_info = get_debug_info()
                raise AssertionError(f"{e}\n\nDEBUG INFO:\n{debug_info}")
            
            text_2 = tokenizer.decode(ids)
            try:
                self.assertIsInstance(text_2, str)
            except AssertionError as e:
                debug_info = get_debug_info()
                raise AssertionError(f"{e}\n\nDEBUG INFO:\n{debug_info}")

            try:
                self.assertEqual(text_2, output_text)
            except AssertionError as e:
                debug_info = get_debug_info()
                raise AssertionError(f"{e}\n\nDEBUG INFO:\n{debug_info}")
        except AssertionError:
            # Re-raise with debug info already included
            raise
        except Exception as e:
            # For non-AssertionError exceptions, wrap with debug info
            debug_info = get_debug_info()
            raise AssertionError(f"Unexpected error: {e}\n\nDEBUG INFO:\n{debug_info}") from e
