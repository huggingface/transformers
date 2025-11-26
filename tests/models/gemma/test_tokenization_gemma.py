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

    @classmethod
    def setUpClass(cls):
        """Override to add debug logging for CI."""
        import os
        import sys
        
        # Store debug info as class variable so we can include it in test failures
        cls._setup_debug_info = []
        
        try:
            print("\n" + "="*80, file=sys.stderr)
            print("[GEMMA TEST DEBUG] Starting setUpClass", file=sys.stderr)
            print("="*80, file=sys.stderr)
            
            # Call parent setUpClass
            super().setUpClass()
            
            # Log what files are in tmpdirname after setup
            if hasattr(cls, 'tmpdirname') and os.path.isdir(cls.tmpdirname):
                files = os.listdir(cls.tmpdirname)
                msg = f"Files in tmpdirname after setup: {sorted(files)}"
                print(f"[GEMMA TEST DEBUG] {msg}", file=sys.stderr)
                cls._setup_debug_info.append(msg)
                
                # Check if tokenizer.json exists and its size
                tokenizer_json = os.path.join(cls.tmpdirname, 'tokenizer.json')
                if os.path.exists(tokenizer_json):
                    size = os.path.getsize(tokenizer_json)
                    msg = f"tokenizer.json exists, size: {size} bytes"
                    print(f"[GEMMA TEST DEBUG] {msg}", file=sys.stderr)
                    cls._setup_debug_info.append(msg)
                    
                    # Check vocab size in the file
                    try:
                        import json as json_module
                        with open(tokenizer_json) as f:
                            tj = json_module.load(f)
                        vocab_size = len(tj.get('model', {}).get('vocab', {}))
                        msg = f"Vocab size in tokenizer.json: {vocab_size}"
                        print(f"[GEMMA TEST DEBUG] {msg}", file=sys.stderr)
                        cls._setup_debug_info.append(msg)
                    except Exception as e:
                        msg = f"Error reading tokenizer.json: {e}"
                        print(f"[GEMMA TEST DEBUG] {msg}", file=sys.stderr)
                        cls._setup_debug_info.append(msg)
                else:
                    msg = "tokenizer.json does NOT exist!"
                    print(f"[GEMMA TEST DEBUG] {msg}", file=sys.stderr)
                    cls._setup_debug_info.append(msg)
                    
                    # Try to manually save the tokenizer to see what happens
                    try:
                        from transformers import AutoTokenizer
                        print(f"[GEMMA TEST DEBUG] Attempting to manually save tokenizer...", file=sys.stderr)
                        tokenizer = AutoTokenizer.from_pretrained(
                            cls.from_pretrained_id[0],
                            **(cls.from_pretrained_kwargs if cls.from_pretrained_kwargs is not None else {}),
                        )
                        print(f"[GEMMA TEST DEBUG] Loaded tokenizer, vocab size: {len(tokenizer)}", file=sys.stderr)
                        tokenizer.save_pretrained(cls.tmpdirname)
                        files_after = os.listdir(cls.tmpdirname)
                        print(f"[GEMMA TEST DEBUG] Files after manual save: {sorted(files_after)}", file=sys.stderr)
                        cls._setup_debug_info.append(f"Files after manual save: {sorted(files_after)}")
                    except Exception as e:
                        msg = f"Error in manual save attempt: {e}"
                        print(f"[GEMMA TEST DEBUG] {msg}", file=sys.stderr)
                        import traceback
                        print(f"[GEMMA TEST DEBUG] Traceback: {traceback.format_exc()}", file=sys.stderr)
                        cls._setup_debug_info.append(msg)
            else:
                msg = "tmpdirname not set or not a directory"
                print(f"[GEMMA TEST DEBUG] {msg}", file=sys.stderr)
                cls._setup_debug_info.append(msg)
            
            print("="*80 + "\n", file=sys.stderr)
        except Exception as e:
            import traceback
            print(f"[GEMMA TEST DEBUG] Error in setUpClass: {e}", file=sys.stderr)
            print(f"[GEMMA TEST DEBUG] Traceback: {traceback.format_exc()}", file=sys.stderr)
            cls._setup_debug_info.append(f"Error in setUpClass: {e}")

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
        tokenizer = self.get_tokenizer()
        
        def get_debug_info():
            """Build debug info string to include in error messages."""
            debug_lines = []
            
            # Include setup debug info
            if hasattr(self.__class__, '_setup_debug_info'):
                debug_lines.append("=== SETUP DEBUG INFO ===")
                debug_lines.extend(self.__class__._setup_debug_info)
                debug_lines.append("")
            
            debug_lines.append("=== TOKENIZER DEBUG INFO ===")
            debug_lines.append(f"Tokenizer type: {type(tokenizer).__name__}")
            debug_lines.append(f"Tokenizer module: {type(tokenizer).__module__}")
            debug_lines.append(f"Tokenizer class location: {tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}")
            
            if hasattr(tokenizer, 'name_or_path'):
                debug_lines.append(f"Tokenizer name_or_path: {tokenizer.name_or_path}")
            if hasattr(tokenizer, 'vocab_file'):
                debug_lines.append(f"Tokenizer vocab_file: {tokenizer.vocab_file}")
            if hasattr(tokenizer, 'vocab_files_names'):
                debug_lines.append(f"Tokenizer vocab_files_names: {tokenizer.vocab_files_names}")
            
            # Check what files exist in the directory
            import os
            if hasattr(tokenizer, 'name_or_path') and os.path.isdir(tokenizer.name_or_path):
                files_in_dir = os.listdir(tokenizer.name_or_path)
                debug_lines.append(f"Files in tokenizer directory: {sorted(files_in_dir)}")
            
            if hasattr(tokenizer, '_tokenizer'):
                debug_lines.append(f"Has _tokenizer attribute: True")
                debug_lines.append(f"_tokenizer type: {type(tokenizer._tokenizer)}")
                if hasattr(tokenizer._tokenizer, 'model'):
                    debug_lines.append(f"_tokenizer.model type: {type(tokenizer._tokenizer.model)}")
                # Try to get more details about the tokenizer
                try:
                    debug_lines.append(f"_tokenizer object: {repr(tokenizer._tokenizer)}")
                except:
                    debug_lines.append(f"_tokenizer object: <repr failed>")
            else:
                debug_lines.append(f"Has _tokenizer attribute: False")
            
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
