# Copyright 2023 The HuggingFace Team. All rights reserved.
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

import copy
import unittest

from transformers import AutoTokenizer, NougatTokenizer
from transformers.models.nougat.tokenization_nougat import markdown_compatible, normalize_list_like_lines
from transformers.testing_utils import require_levenshtein, require_nltk, require_tokenizers

from ...test_tokenization_common import TokenizerTesterMixin



@require_tokenizers
class NougatTokenizationTest(TokenizerTesterMixin, unittest.TestCase):
    from_pretrained_id = "facebook/nougat-base"
    slow_tokenizer_class = None
    tokenizer_class = NougatTokenizer
    test_slow_tokenizer = False
    from_pretrained_vocab_key = "tokenizer_file"
    special_tokens_map = {"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>", "pad_token": "<pad>"}


    # Integration test data - expected outputs for the default input string
    integration_expected_tokens = ['This', 'Ġis', 'Ġa', 'Ġtest', 'Ċ', 'I', 'Ġwas', 'Ġborn', 'Ġin', 'Ġ', '9', '2', '0', '0', '0', ',', 'Ġand', 'Ġthis', 'Ġis', 'Ġfals', 'Ã©', '.', 'Ċ', 'çĶ', 'Ł', 'æ', '´', '»', 'çļĦ', 'ç', 'ľ', 'Ł', 'è', '°', 'Ľ', 'æ', 'ĺ', '¯', 'Ċ', 'Hi', 'Ġ', 'ĠH', 'ello', 'Ċ', 'Hi', 'ĠĠ', 'ĠH', 'ello', 'Ċ', 'Ċ', 'Ġ', 'Ċ', 'ĠĠ', 'Ċ', 'ĠH', 'ello', 'Ċ', '<s>', 'Ċ', 'hi', '<s>', 'there', 'Ċ', 'The', 'Ġfollowing', 'Ġstring', 'Ġshould', 'Ġbe', 'Ġproperly', 'Ġencoded', ':', 'ĠH', 'ello', '.', 'Ċ', 'But', 'Ġ', 'ird', 'Ġand', 'Ġ', 'à¸', 'Ľ', 'à¸', 'µ', 'ĠĠ', 'Ġ', 'ird', 'ĠĠ', 'Ġ', 'à¸', 'Ķ', 'Ċ', 'H', 'ey', 'Ġhow', 'Ġare', 'Ġyou', 'Ġdoing']
    integration_expected_token_ids = [0, 2113, 343, 281, 1185, 221, 63, 435, 8613, 301, 243, 47, 40, 38, 38, 38, 34, 312, 495, 343, 34500, 2230, 36, 221, 33239, 276, 185, 135, 142, 31778, 186, 273, 276, 187, 131, 272, 185, 269, 130, 221, 33719, 243, 414, 13716, 221, 33719, 304, 414, 13716, 221, 221, 243, 221, 304, 221, 414, 13716, 221, 0, 221, 2197, 0, 10158, 221, 592, 1093, 4935, 1502, 391, 10651, 10033, 48, 414, 13716, 36, 221, 11847, 243, 2326, 312, 243, 12043, 272, 12043, 136, 304, 243, 2326, 304, 243, 12043, 265, 221, 62, 1220, 1905, 417, 2589, 10671, 2]
    integration_expected_decoded_text = 'This is a test 😊\nI was born in 92000, and this is falsé.\n生活的真谛是\nHi  Hello\nHi   Hello\n\n \n  \n Hello\n<s>\nhi<s>there\nThe following string should be properly encoded: Hello.\nBut ird and ปี   ird   ด\nHey how are you doing'
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        
        tokenizer = NougatTokenizer.from_pretrained("facebook/nougat-base")
        tokenizer.save_pretrained(cls.tmpdirname)

    def test_padding(self, max_length=6):
        for tokenizer, pretrained_name, kwargs in self.tokenizers_list:
            with self.subTest(f"{tokenizer.__class__.__name__} ({pretrained_name})"):
                tokenizer_r = self.get_rust_tokenizer(pretrained_name, **kwargs)
                # Simple input
                sentence1 = "This is a simple input"
                sentence2 = ["This is a simple input 1", "This is a simple input 2"]
                pair1 = ("This is a simple input", "This is a pair")
                pair2 = [
                    ("This is a simple input 1", "This is a simple input 2"),
                    ("This is a simple pair 1", "This is a simple pair 2"),
                ]

                # Simple input tests
                try:
                    tokenizer_r.encode(sentence1, max_length=max_length)
                    tokenizer_r(sentence1, max_length=max_length)

                    tokenizer_r(sentence2, max_length=max_length)
                    tokenizer_r.encode(pair1, max_length=max_length)
                    tokenizer_r(pair2, max_length=max_length)
                except ValueError:
                    self.fail("Nougat Tokenizer should be able to deal with padding")

                tokenizer_r.pad_token = None  # Hotfixing padding = None
                self.assertRaises(
                    ValueError, tokenizer_r.encode, sentence1, max_length=max_length, padding="max_length"
                )

                # Simple input
                self.assertRaises(
                    ValueError, tokenizer_r, sentence1, max_length=max_length, padding="max_length"
                )

                # Simple input
                self.assertRaises(
                    ValueError,
                    tokenizer_r,
                    sentence2,
                    max_length=max_length,
                    padding="max_length",
                )

                # Pair input
                self.assertRaises(ValueError, tokenizer_r.encode, pair1, max_length=max_length, padding="max_length")

                # Pair input
                self.assertRaises(
                    ValueError, tokenizer_r, pair1, max_length=max_length, padding="max_length"
                )

                # Pair input
                self.assertRaises(
                    ValueError,
                    tokenizer_r,
                    pair2,
                    max_length=max_length,
                    padding="max_length",
                )

class MarkdownCompatibleTest(unittest.TestCase):
    def test_equation_tag(self):
        input_text = "(3.2) \\[Equation Text\\]"
        excepted_output = "\\[Equation Text \\tag{3.2}\\]"
        self.assertEqual(markdown_compatible(input_text), excepted_output)

    def test_equation_tag_letters(self):
        input_text = "(18a) \\[Equation Text\\]"
        excepted_output = "\\[Equation Text \\tag{18a}\\]"
        self.assertEqual(markdown_compatible(input_text), excepted_output)

    def test_bold_formatting(self):
        input_text = r"This is \bm{bold} text."
        expected_output = r"This is \mathbf{bold} text."
        self.assertEqual(markdown_compatible(input_text), expected_output)

    def test_url_conversion(self):
        input_text = "Visit my website at https://www.example.com"
        expected_output = "Visit my website at [https://www.example.com](https://www.example.com)"
        self.assertEqual(markdown_compatible(input_text), expected_output)

    def test_algorithm_code_block(self):
        input_text = "```python\nprint('Hello, world!')\n```"
        expected_output = "```\npython\nprint('Hello, world!')\n```"
        self.assertEqual(markdown_compatible(input_text), expected_output)

    def test_escape_characters(self):
        input_text = r"Escaped characters like \n should not be \\[affected\\]"
        expected_output = r"Escaped characters like \n should not be \\[affected\\]"
        self.assertEqual(markdown_compatible(input_text), expected_output)

    def test_nested_tags(self):
        input_text = r"This is a super nested \bm{\bm{\bm{\bm{\bm{bold}}}}} tag."
        expected_output = r"This is a super nested \mathbf{\mathbf{\mathbf{\mathbf{\mathbf{bold}}}}} tag."
        self.assertEqual(markdown_compatible(input_text), expected_output)


class TestNormalizeListLikeLines(unittest.TestCase):
    def test_two_level_lines(self):
        input_str = "* Item 1 * Item 2"
        expected_output = "* Item 1\n* Item 2\n"
        self.assertEqual(normalize_list_like_lines(input_str), expected_output)

    def test_three_level_lines(self):
        input_str = "- I. Item 1 - II. Item 2 - III. Item 3"
        expected_output = "- I. Item 1\n- II. Item 2\n- III. Item 3\n"
        self.assertEqual(normalize_list_like_lines(input_str), expected_output)

    def test_nested_lines(self):
        input_str = "- I. Item 1 - I.1 Sub-item 1 - I.1.1 Sub-sub-item 1 - II. Item 2"
        expected_output = "- I. Item 1\n\t- I.1 Sub-item 1\n\t\t- I.1.1 Sub-sub-item 1\n- II. Item 2\n"
        self.assertEqual(normalize_list_like_lines(input_str), expected_output)


@require_tokenizers
class NougatPostProcessingTest(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.tokenizer = NougatTokenizer.from_pretrained("facebook/nougat-base")

    def test_correct_tables_basic(self):
        input_str = "\\begin{table} \\begin{tabular}{l l}  & \\ \\end{tabular} \\end{table}"
        expected_output = "\\begin{table}\n\\begin{tabular}{l l}  & \\ \\end{tabular}\n\\end{table}"
        self.assertEqual(self.tokenizer.correct_tables(input_str), expected_output)

    def test_correct_tables_high_count(self):
        input_str = "\\begin{tabular}" * 20
        expected_output = ""
        self.assertEqual(self.tokenizer.correct_tables(input_str), expected_output)

    @require_levenshtein
    @require_nltk
    def test_postprocess_as_nougat_no_markdown(self):
        input_str = "# Nougat: Neural Optical Understanding for Academic Documents\n\n Lukas Blecher\n\nCorrespondence to: lblecher@meta.com\n\nGuillem Cucurull\n\nThomas Scialom\n\nRobert Stojnic\n\nMeta AI\n\nThe paper reports 8.1M papers but the authors recently updated the numbers on the GitHub page https://github.com/allenai/s2orc\n\n###### Abstract\n\nScientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (**N**eural **O**ptical **U**nderstanding for **A**cademic Documents), a Visual Transformer model that performs an _Optical Character Recognition_ (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the accessibility of scientific knowledge in the digital age, by bridging the gap between human-readable documents and machine-readable text. We release the models and code to accelerate future work on scientific text recognition.\n\n## 1 Introduction\n\nThe majority of scientific knowledge is stored in books or published in scientific journals, most commonly in the Portable Document Format (PDF). Next to HTML, PDFs are the second most prominent data format on the internet, making up 2.4% of common crawl [1]. However, the information stored in these files is very difficult to extract into any other formats. This is especially true for highly specialized documents, such as scientific research papers, where the semantic information of mathematical expressions is lost.\n\nExisting Optical Character Recognition (OCR) engines, such as Tesseract OCR [2], excel at detecting and classifying individual characters and words in an image, but fail to understand the relationship between them due to their line-by-line approach. This means that they treat superscripts and subscripts in the same way as the surrounding text, which is a significant drawback for mathematical expressions. In mathematical notations like fractions, exponents, and matrices, relative positions of characters are crucial.\n\nConverting academic research papers into machine-readable text also enables accessibility and searchability of science as a whole. The information of millions of academic papers can not be fully accessed because they are locked behind an unreadable format. Existing corpora, such as the S2ORC dataset [3], capture the text of 12M2 papers using GROBID [4], but are missing meaningful representations of the mathematical equations.\n\nFootnote 2: The paper reports 8.1M papers but the authors recently updated the numbers on the GitHub page https://github.com/allenai/s2orc\n\nTo this end, we introduce Nougat, a transformer based model that can convert images of document pages to formatted markup text.\n\nThe primary contributions in this paper are\n\n* Release of a pre-trained model capable of converting a PDF to a lightweight markup language. We release the code and the model on GitHub3 Footnote 3: https://github.com/facebookresearch/nougat\n* We introduce a pipeline to create dataset for pairing PDFs to source code\n* Our method is only dependent on the image of a page, allowing access to scanned papers and books"  # noqa: E231
        expected_output = "\n\n# Nougat: Neural Optical Understanding for Academic Documents\n\n Lukas Blecher\n\nCorrespondence to: lblecher@meta.com\n\nGuillem Cucurull\n\nThomas Scialom\n\nRobert Stojnic\n\nMeta AI\n\nThe paper reports 8.1M papers but the authors recently updated the numbers on the GitHub page https://github.com/allenai/s2orc\n\n###### Abstract\n\nScientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (**N**eural **O**ptical **U**nderstanding for **A**cademic Documents), a Visual Transformer model that performs an _Optical Character Recognition_ (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the accessibility of scientific knowledge in the digital age, by bridging the gap between human-readable documents and machine-readable text. We release the models and code to accelerate future work on scientific text recognition.\n\n## 1 Introduction\n\nThe majority of scientific knowledge is stored in books or published in scientific journals, most commonly in the Portable Document Format (PDF). Next to HTML, PDFs are the second most prominent data format on the internet, making up 2.4% of common crawl [1]. However, the information stored in these files is very difficult to extract into any other formats. This is especially true for highly specialized documents, such as scientific research papers, where the semantic information of mathematical expressions is lost.\n\nExisting Optical Character Recognition (OCR) engines, such as Tesseract OCR [2], excel at detecting and classifying individual characters and words in an image, but fail to understand the relationship between them due to their line-by-line approach. This means that they treat superscripts and subscripts in the same way as the surrounding text, which is a significant drawback for mathematical expressions. In mathematical notations like fractions, exponents, and matrices, relative positions of characters are crucial.\n\nConverting academic research papers into machine-readable text also enables accessibility and searchability of science as a whole. The information of millions of academic papers can not be fully accessed because they are locked behind an unreadable format. Existing corpora, such as the S2ORC dataset [3], capture the text of 12M2 papers using GROBID [4], but are missing meaningful representations of the mathematical equations.\n\nFootnote 2: The paper reports 8.1M papers but the authors recently updated the numbers on the GitHub page https://github.com/allenai/s2orc\n\nTo this end, we introduce Nougat, a transformer based model that can convert images of document pages to formatted markup text.\n\nThe primary contributions in this paper are\n\n* Release of a pre-trained model capable of converting a PDF to a lightweight markup language. We release the code and the model on GitHub3 Footnote 3: https://github.com/facebookresearch/nougat\n* We introduce a pipeline to create dataset for pairing PDFs to source code\n* Our method is only dependent on the image of a page, allowing access to scanned papers and books"  # noqa: E231
        self.assertEqual(self.tokenizer.post_process_single(input_str, fix_markdown=False), expected_output)
