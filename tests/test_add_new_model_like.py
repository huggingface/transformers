# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import os
import re
import tempfile
import unittest
from pathlib import Path

import transformers
from transformers.commands.add_new_model_like import (
    ModelPatterns,
    _re_class_func,
    add_content_to_file,
    add_content_to_text,
    clean_frameworks_in_init,
    duplicate_doc_file,
    duplicate_module,
    filter_framework_files,
    find_base_model_checkpoint,
    get_model_files,
    get_module_from_file,
    parse_module_content,
    replace_model_patterns,
    retrieve_info_for_model,
    retrieve_model_classes,
)
from transformers.testing_utils import require_flax, require_tf, require_torch


BERT_MODEL_FILES = {
    "src/transformers/models/bert/__init__.py",
    "src/transformers/models/bert/configuration_bert.py",
    "src/transformers/models/bert/tokenization_bert.py",
    "src/transformers/models/bert/tokenization_bert_fast.py",
    "src/transformers/models/bert/modeling_bert.py",
    "src/transformers/models/bert/modeling_flax_bert.py",
    "src/transformers/models/bert/modeling_tf_bert.py",
    "src/transformers/models/bert/convert_bert_original_tf_checkpoint_to_pytorch.py",
    "src/transformers/models/bert/convert_bert_original_tf2_checkpoint_to_pytorch.py",
    "src/transformers/models/bert/convert_bert_pytorch_checkpoint_to_original_tf.py",
}

REPO_PATH = Path(transformers.__path__[0]).parent.parent


@require_torch
@require_tf
@require_flax
class TestAddNewModelLike(unittest.TestCase):
    def init_file(self, file_name, content):
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(content)

    def check_result(self, file_name, expected_result):
        with open(file_name, "r", encoding="utf-8") as f:
            self.assertEqual(f.read(), expected_result)

    def test_re_class_func(self):
        self.assertEqual(_re_class_func.search("def my_function(x, y):").groups()[0], "my_function")
        self.assertEqual(_re_class_func.search("class MyClass:").groups()[0], "MyClass")
        self.assertEqual(_re_class_func.search("class MyClass(SuperClass):").groups()[0], "MyClass")

    def test_model_patterns_defaults(self):
        model_patterns = ModelPatterns("GPT-New new", "huggingface/gpt-new-base")

        self.assertEqual(model_patterns.model_type, "gpt-new-new")
        self.assertEqual(model_patterns.model_lower_cased, "gpt_new_new")
        self.assertEqual(model_patterns.model_camel_cased, "GPTNewNew")
        self.assertEqual(model_patterns.model_upper_cased, "GPT_NEW_NEW")
        self.assertEqual(model_patterns.tokenizer_class, "GPTNewNewTokenizer")
        self.assertEqual(model_patterns.config_class, "GPTNewNewConfig")

    def test_parse_module_content(self):
        test_code = """SOME_CONSTANT = a constant

CONSTANT_DEFINED_ON_SEVERAL_LINES = [
    first_item,
    second_item
]

def function(args):
    some code

class SomeClass:
    some code
"""

        expected_parts = [
            "SOME_CONSTANT = a constant\n",
            "CONSTANT_DEFINED_ON_SEVERAL_LINES = [\n    first_item,\n    second_item\n]",
            "",
            "def function(args):\n    some code\n",
            "class SomeClass:\n    some code\n",
        ]
        self.assertEqual(parse_module_content(test_code), expected_parts)

    def test_add_content_to_text(self):
        test_text = """all_configs = {
    "gpt": "GPTConfig",
    "bert": "BertConfig",
    "t5": "T5Config",
}"""

        expected = """all_configs = {
    "gpt": "GPTConfig",
    "gpt2": "GPT2Config",
    "bert": "BertConfig",
    "t5": "T5Config",
}"""
        line = '    "gpt2": "GPT2Config",'

        self.assertEqual(add_content_to_text(test_text, line, add_before="bert"), expected)
        self.assertEqual(add_content_to_text(test_text, line, add_before="bert", exact_match=True), test_text)
        self.assertEqual(
            add_content_to_text(test_text, line, add_before='    "bert": "BertConfig",', exact_match=True), expected
        )
        self.assertEqual(add_content_to_text(test_text, line, add_before=re.compile('^\s*"bert":')), expected)

        self.assertEqual(add_content_to_text(test_text, line, add_after="gpt"), expected)
        self.assertEqual(add_content_to_text(test_text, line, add_after="gpt", exact_match=True), test_text)
        self.assertEqual(
            add_content_to_text(test_text, line, add_after='    "gpt": "GPTConfig",', exact_match=True), expected
        )
        self.assertEqual(add_content_to_text(test_text, line, add_after=re.compile('^\s*"gpt":')), expected)

    def test_add_content_to_file(self):
        test_text = """all_configs = {
    "gpt": "GPTConfig",
    "bert": "BertConfig",
    "t5": "T5Config",
}"""

        expected = """all_configs = {
    "gpt": "GPTConfig",
    "gpt2": "GPT2Config",
    "bert": "BertConfig",
    "t5": "T5Config",
}"""
        line = '    "gpt2": "GPT2Config",'

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, "code.py")

            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_before="bert")
            self.check_result(file_name, expected)

            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_before="bert", exact_match=True)
            self.check_result(file_name, test_text)

            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_before='    "bert": "BertConfig",', exact_match=True)
            self.check_result(file_name, expected)

            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_before=re.compile('^\s*"bert":'))
            self.check_result(file_name, expected)

            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_after="gpt")
            self.check_result(file_name, expected)

            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_after="gpt", exact_match=True)
            self.check_result(file_name, test_text)

            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_after='    "gpt": "GPTConfig",', exact_match=True)
            self.check_result(file_name, expected)

            self.init_file(file_name, test_text)
            add_content_to_file(file_name, line, add_after=re.compile('^\s*"gpt":'))
            self.check_result(file_name, expected)

    def test_replace_model_patterns(self):
        bert_model_patterns = ModelPatterns("Bert", "bert-base-cased")
        new_bert_model_patterns = ModelPatterns("New Bert", "huggingface/bert-new-base")
        bert_test = '''class TFBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    model_type = "bert"

BERT_CONSTANT = "value"
'''
        bert_expected = '''class TFNewBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = NewBertConfig
    load_tf_weights = load_tf_weights_in_new_bert
    base_model_prefix = "new_bert"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    model_type = "new-bert"

NEW_BERT_CONSTANT = "value"
'''

        bert_converted, replacements = replace_model_patterns(bert_test, bert_model_patterns, new_bert_model_patterns)
        self.assertEqual(bert_converted, bert_expected)
        # TODO: model type replacement missing here
        self.assertEqual(replacements, "bert->new_bert,Bert->NewBert,BERT->NEW_BERT")

        gpt_model_patterns = ModelPatterns("GPT2", "gpt2")
        new_gpt_model_patterns = ModelPatterns("GPT-New new", "huggingface/gpt-new-base")
        gpt_test = '''class GPT2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPT2Config
    load_tf_weights = load_tf_weights_in_gpt2
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

GPT2_CONSTANT = "value"
'''

        gpt_expected = '''class GPTNewNewPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = GPTNewNewConfig
    load_tf_weights = load_tf_weights_in_gpt_new_new
    base_model_prefix = "transformer"
    is_parallelizable = True
    supports_gradient_checkpointing = True

GPT_NEW_NEW_CONSTANT = "value"
'''

        gpt_converted, replacements = replace_model_patterns(gpt_test, gpt_model_patterns, new_gpt_model_patterns)
        self.assertEqual(gpt_converted, gpt_expected)
        # Replacements are empty here since GPT2 as been replaced by GPTNewNew in some instances and GPT_NEW_NEW
        # in others.
        self.assertEqual(replacements, "")

    def test_get_module_from_file(self):
        self.assertEqual(
            get_module_from_file("/git/transformers/src/transformers/models/bert/modeling_tf_bert.py"),
            "transformers.models.bert.modeling_tf_bert",
        )
        self.assertEqual(
            get_module_from_file("/transformers/models/gpt2/modeling_gpt2.py"),
            "transformers.models.gpt2.modeling_gpt2",
        )
        with self.assertRaises(ValueError):
            get_module_from_file("/models/gpt2/modeling_gpt2.py")

    def test_duplicate_module(self):
        bert_model_patterns = ModelPatterns("Bert", "bert-base-cased")
        new_bert_model_patterns = ModelPatterns("New Bert", "huggingface/bert-new-base")
        bert_test = '''class TFBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    model_type = "bert"

BERT_CONSTANT = "value"
'''
        bert_expected = '''class TFNewBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = NewBertConfig
    load_tf_weights = load_tf_weights_in_new_bert
    base_model_prefix = "new_bert"
    is_parallelizable = True
    supports_gradient_checkpointing = True
    model_type = "new-bert"

NEW_BERT_CONSTANT = "value"
'''
        bert_expected_with_copied_from = (
            "# Copied from transformers.bert_module.TFBertPreTrainedModel with bert->new_bert,Bert->NewBert\n"
            + bert_expected
        )
        with tempfile.TemporaryDirectory() as tmp_dir:
            work_dir = os.path.join(tmp_dir, "transformers")
            os.makedirs(work_dir)
            file_name = os.path.join(work_dir, "bert_module.py")
            dest_file_name = os.path.join(work_dir, "new_bert_module.py")

            self.init_file(file_name, bert_test)
            duplicate_module(file_name, bert_model_patterns, new_bert_model_patterns)
            self.check_result(dest_file_name, bert_expected_with_copied_from)

            self.init_file(file_name, bert_test)
            duplicate_module(file_name, bert_model_patterns, new_bert_model_patterns, add_copied_from=False)
            self.check_result(dest_file_name, bert_expected)

    def test_filter_framework_files(self):
        files = ["modeling_tf_bert.py", "modeling_bert.py", "modeling_flax_bert.py", "configuration_bert.py"]
        self.assertEqual(filter_framework_files(files), files)
        self.assertEqual(set(filter_framework_files(files, ["pt", "tf", "flax"])), set(files))

        self.assertEqual(set(filter_framework_files(files, ["pt"])), {"modeling_bert.py", "configuration_bert.py"})
        self.assertEqual(set(filter_framework_files(files, ["tf"])), {"modeling_tf_bert.py", "configuration_bert.py"})
        self.assertEqual(
            set(filter_framework_files(files, ["flax"])), {"modeling_flax_bert.py", "configuration_bert.py"}
        )

        self.assertEqual(
            set(filter_framework_files(files, ["pt", "tf"])),
            {"modeling_tf_bert.py", "modeling_bert.py", "configuration_bert.py"},
        )
        self.assertEqual(
            set(filter_framework_files(files, ["tf", "flax"])),
            {"modeling_tf_bert.py", "modeling_flax_bert.py", "configuration_bert.py"},
        )
        self.assertEqual(
            set(filter_framework_files(files, ["pt", "flax"])),
            {"modeling_bert.py", "modeling_flax_bert.py", "configuration_bert.py"},
        )

    def test_get_model_files(self):
        bert_files = get_model_files("bert")

        doc_file = str(Path(bert_files["doc_file"]).relative_to(REPO_PATH))
        self.assertEqual(doc_file, "docs/source/model_doc/bert.mdx")

        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files["model_files"]}
        self.assertEqual(model_files, BERT_MODEL_FILES)

        self.assertEqual(bert_files["module_name"], "bert")

        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files["test_files"]}
        bert_test_files = {
            "tests/test_tokenization_bert.py",
            "tests/test_modeling_bert.py",
            "tests/test_modeling_tf_bert.py",
            "tests/test_modeling_flax_bert.py",
        }
        self.assertEqual(test_files, bert_test_files)

    def test_get_model_files_only_pt(self):
        bert_files = get_model_files("bert", frameworks=["pt"])

        doc_file = str(Path(bert_files["doc_file"]).relative_to(REPO_PATH))
        self.assertEqual(doc_file, "docs/source/model_doc/bert.mdx")

        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files["model_files"]}
        bert_model_files = BERT_MODEL_FILES - {
            "src/transformers/models/bert/modeling_tf_bert.py",
            "src/transformers/models/bert/modeling_flax_bert.py",
        }
        self.assertEqual(model_files, bert_model_files)

        self.assertEqual(bert_files["module_name"], "bert")

        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files["test_files"]}
        bert_test_files = {
            "tests/test_tokenization_bert.py",
            "tests/test_modeling_bert.py",
        }
        self.assertEqual(test_files, bert_test_files)

    def test_get_model_files_tf_and_flax(self):
        bert_files = get_model_files("bert", frameworks=["tf", "flax"])

        doc_file = str(Path(bert_files["doc_file"]).relative_to(REPO_PATH))
        self.assertEqual(doc_file, "docs/source/model_doc/bert.mdx")

        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files["model_files"]}
        bert_model_files = BERT_MODEL_FILES - {"src/transformers/models/bert/modeling_bert.py"}
        self.assertEqual(model_files, bert_model_files)

        self.assertEqual(bert_files["module_name"], "bert")

        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in bert_files["test_files"]}
        bert_test_files = {
            "tests/test_tokenization_bert.py",
            "tests/test_modeling_tf_bert.py",
            "tests/test_modeling_flax_bert.py",
        }
        self.assertEqual(test_files, bert_test_files)

    def test_find_base_model_checkpoint(self):
        self.assertEqual(find_base_model_checkpoint("bert"), "bert-base-uncased")
        self.assertEqual(find_base_model_checkpoint("gpt2"), "gpt2")

    def test_retrieve_model_classes(self):
        gpt_classes = {k: set(v) for k, v in retrieve_model_classes("gpt2").items()}
        expected_gpt_classes = {
            "pt": {"GPT2ForTokenClassification", "GPT2Model", "GPT2LMHeadModel", "GPT2ForSequenceClassification"},
            "tf": {"TFGPT2Model", "TFGPT2ForSequenceClassification", "TFGPT2LMHeadModel"},
            "flax": {"FlaxGPT2Model", "FlaxGPT2LMHeadModel"},
        }
        self.assertEqual(gpt_classes, expected_gpt_classes)

        del expected_gpt_classes["flax"]
        gpt_classes = {k: set(v) for k, v in retrieve_model_classes("gpt2", frameworks=["pt", "tf"]).items()}
        self.assertEqual(gpt_classes, expected_gpt_classes)

        del expected_gpt_classes["pt"]
        gpt_classes = {k: set(v) for k, v in retrieve_model_classes("gpt2", frameworks=["tf"]).items()}
        self.assertEqual(gpt_classes, expected_gpt_classes)

    def test_retrieve_info_for_model_with_bert(self):
        bert_info = retrieve_info_for_model("bert")
        bert_classes = [
            "BertForTokenClassification",
            "BertForQuestionAnswering",
            "BertForNextSentencePrediction",
            "BertForSequenceClassification",
            "BertForMaskedLM",
            "BertForMultipleChoice",
            "BertModel",
            "BertForPreTraining",
            "BertLMHeadModel",
        ]
        expected_model_classes = {
            "pt": set(bert_classes),
            "tf": {f"TF{m}" for m in bert_classes},
            "flax": {f"Flax{m}" for m in bert_classes[:-1]},
        }

        self.assertEqual(set(bert_info["frameworks"]), {"pt", "tf", "flax"})
        model_classes = {k: set(v) for k, v in bert_info["model_classes"].items()}
        self.assertEqual(model_classes, expected_model_classes)

        all_bert_files = bert_info["model_files"]
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_bert_files["model_files"]}
        self.assertEqual(model_files, BERT_MODEL_FILES)

        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_bert_files["test_files"]}
        bert_test_files = {
            "tests/test_tokenization_bert.py",
            "tests/test_modeling_bert.py",
            "tests/test_modeling_tf_bert.py",
            "tests/test_modeling_flax_bert.py",
        }
        self.assertEqual(test_files, bert_test_files)

        doc_file = str(Path(all_bert_files["doc_file"]).relative_to(REPO_PATH))
        self.assertEqual(doc_file, "docs/source/model_doc/bert.mdx")

        self.assertEqual(all_bert_files["module_name"], "bert")

        bert_model_patterns = bert_info["model_patterns"]
        self.assertEqual(bert_model_patterns.model_name, "BERT")
        self.assertEqual(bert_model_patterns.checkpoint, "bert-base-uncased")
        self.assertEqual(bert_model_patterns.model_type, "bert")
        self.assertEqual(bert_model_patterns.model_lower_cased, "bert")
        self.assertEqual(bert_model_patterns.model_camel_cased, "Bert")
        self.assertEqual(bert_model_patterns.model_upper_cased, "BERT")
        self.assertEqual(bert_model_patterns.config_class, "BertConfig")
        self.assertEqual(bert_model_patterns.tokenizer_class, "BertTokenizer")

    def test_retrieve_info_for_model_pt_tf_with_bert(self):
        bert_info = retrieve_info_for_model("bert", frameworks=["pt", "tf"])
        bert_classes = [
            "BertForTokenClassification",
            "BertForQuestionAnswering",
            "BertForNextSentencePrediction",
            "BertForSequenceClassification",
            "BertForMaskedLM",
            "BertForMultipleChoice",
            "BertModel",
            "BertForPreTraining",
            "BertLMHeadModel",
        ]
        expected_model_classes = {"pt": set(bert_classes), "tf": {f"TF{m}" for m in bert_classes}}

        self.assertEqual(set(bert_info["frameworks"]), {"pt", "tf"})
        model_classes = {k: set(v) for k, v in bert_info["model_classes"].items()}
        self.assertEqual(model_classes, expected_model_classes)

        all_bert_files = bert_info["model_files"]
        model_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_bert_files["model_files"]}
        bert_model_files = BERT_MODEL_FILES - {"src/transformers/models/bert/modeling_flax_bert.py"}
        self.assertEqual(model_files, bert_model_files)

        test_files = {str(Path(f).relative_to(REPO_PATH)) for f in all_bert_files["test_files"]}
        bert_test_files = {
            "tests/test_tokenization_bert.py",
            "tests/test_modeling_bert.py",
            "tests/test_modeling_tf_bert.py",
        }
        self.assertEqual(test_files, bert_test_files)

        doc_file = str(Path(all_bert_files["doc_file"]).relative_to(REPO_PATH))
        self.assertEqual(doc_file, "docs/source/model_doc/bert.mdx")

        self.assertEqual(all_bert_files["module_name"], "bert")

        bert_model_patterns = bert_info["model_patterns"]
        self.assertEqual(bert_model_patterns.model_name, "BERT")
        self.assertEqual(bert_model_patterns.checkpoint, "bert-base-uncased")
        self.assertEqual(bert_model_patterns.model_type, "bert")
        self.assertEqual(bert_model_patterns.model_lower_cased, "bert")
        self.assertEqual(bert_model_patterns.model_camel_cased, "Bert")
        self.assertEqual(bert_model_patterns.model_upper_cased, "BERT")
        self.assertEqual(bert_model_patterns.config_class, "BertConfig")
        self.assertEqual(bert_model_patterns.tokenizer_class, "BertTokenizer")

    def test_clean_frameworks_in_init(self):
        test_init = """
from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_flax_available, is_tf_available, is_tokenizers_available, is_torch_available

_import_structure = {
    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],
    "tokenization_gpt2": ["GPT2Tokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_gpt2_fast"] = ["GPT2TokenizerFast"]

if is_torch_available():
    _import_structure["modeling_gpt2"] = ["GPT2Model"]

if is_tf_available():
    _import_structure["modeling_tf_gpt2"] = ["TFGPT2Model"]

if is_flax_available():
    _import_structure["modeling_flax_gpt2"] = ["FlaxGPT2Model"]

if TYPE_CHECKING:
    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig
    from .tokenization_gpt2 import GPT2Tokenizer

    if is_tokenizers_available():
        from .tokenization_gpt2_fast import GPT2TokenizerFast

    if is_torch_available():
        from .modeling_gpt2 import GPT2Model

    if is_tf_available():
        from .modeling_tf_gpt2 import TFGPT2Model

    if is_flax_available():
        from .modeling_flax_gpt2 import FlaxGPT2Model

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
"""

        init_no_tokenizer = """
from typing import TYPE_CHECKING

from ...file_utils import _LazyModule, is_flax_available, is_tf_available,  is_torch_available

_import_structure = {
    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],
}

if is_torch_available():
    _import_structure["modeling_gpt2"] = ["GPT2Model"]

if is_tf_available():
    _import_structure["modeling_tf_gpt2"] = ["TFGPT2Model"]

if is_flax_available():
    _import_structure["modeling_flax_gpt2"] = ["FlaxGPT2Model"]

if TYPE_CHECKING:
    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig

    if is_torch_available():
        from .modeling_gpt2 import GPT2Model

    if is_tf_available():
        from .modeling_tf_gpt2 import TFGPT2Model

    if is_flax_available():
        from .modeling_flax_gpt2 import FlaxGPT2Model

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
"""

        init_pt_only = """
from typing import TYPE_CHECKING

from ...file_utils import _LazyModule,   is_tokenizers_available, is_torch_available

_import_structure = {
    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],
    "tokenization_gpt2": ["GPT2Tokenizer"],
}

if is_tokenizers_available():
    _import_structure["tokenization_gpt2_fast"] = ["GPT2TokenizerFast"]

if is_torch_available():
    _import_structure["modeling_gpt2"] = ["GPT2Model"]

if TYPE_CHECKING:
    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig
    from .tokenization_gpt2 import GPT2Tokenizer

    if is_tokenizers_available():
        from .tokenization_gpt2_fast import GPT2TokenizerFast

    if is_torch_available():
        from .modeling_gpt2 import GPT2Model

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
"""

        init_pt_only_no_tokenizer = """
from typing import TYPE_CHECKING

from ...file_utils import _LazyModule,    is_torch_available

_import_structure = {
    "configuration_gpt2": ["GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP", "GPT2Config", "GPT2OnnxConfig"],
}

if is_torch_available():
    _import_structure["modeling_gpt2"] = ["GPT2Model"]

if TYPE_CHECKING:
    from .configuration_gpt2 import GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP, GPT2Config, GPT2OnnxConfig

    if is_torch_available():
        from .modeling_gpt2 import GPT2Model

else:
    import sys

    sys.modules[__name__] = _LazyModule(__name__, globals()["__file__"], _import_structure)
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_name = os.path.join(tmp_dir, "__init__.py")

            self.init_file(file_name, test_init)
            clean_frameworks_in_init(file_name, keep_tokenizer=False)
            self.check_result(file_name, init_no_tokenizer)

            self.init_file(file_name, test_init)
            clean_frameworks_in_init(file_name, frameworks=["pt"])
            self.check_result(file_name, init_pt_only)

            self.init_file(file_name, test_init)
            clean_frameworks_in_init(file_name, frameworks=["pt"], keep_tokenizer=False)
            self.check_result(file_name, init_pt_only_no_tokenizer)

    def test_duplicate_doc_file(self):
        test_doc = """
# GPT2

## Overview

Overview of the model.

## GPT2Config

[[autodoc]] GPT2Config

## GPT2Tokenizer

[[autodoc]] GPT2Tokenizer
    - save_vocabulary

## GPT2TokenizerFast

[[autodoc]] GPT2TokenizerFast

## GPT2 specific outputs

[[autodoc]] models.gpt2.modeling_gpt2.GPT2DoubleHeadsModelOutput

[[autodoc]] models.gpt2.modeling_tf_gpt2.TFGPT2DoubleHeadsModelOutput

## GPT2Model

[[autodoc]] GPT2Model
    - forward

## TFGPT2Model

[[autodoc]] TFGPT2Model
    - call

## FlaxGPT2Model

[[autodoc]] FlaxGPT2Model
    - __call__

"""
        test_new_doc = """
# GPT-New New

## Overview

The GPT-New New model was proposed in [<INSERT PAPER NAME HERE>(<INSERT PAPER LINK HERE>) by <INSERT AUTHORS HERE>.
<INSERT SHORT SUMMARY HERE>

The abstract from the paper is the following:

*<INSERT PAPER ABSTRACT HERE>*

Tips:

<INSERT TIPS ABOUT MODEL HERE>

This model was contributed by [INSERT YOUR HF USERNAME HERE](<https://huggingface.co/<INSERT YOUR HF USERNAME HERE>).
The original code can be found [here](<INSERT LINK TO GITHUB REPO HERE>).


## GPTNewNewConfig

[[autodoc]] GPTNewNewConfig

## GPTNewNewTokenizer

[[autodoc]] GPTNewNewTokenizer
    - save_vocabulary

## GPTNewNewTokenizerFast

[[autodoc]] GPTNewNewTokenizerFast

## GPTNewNew specific outputs

[[autodoc]] models.gpt_new_new.modeling_gpt_new_new.GPTNewNewDoubleHeadsModelOutput

[[autodoc]] models.gpt_new_new.modeling_tf_gpt_new_new.TFGPTNewNewDoubleHeadsModelOutput

## GPTNewNewModel

[[autodoc]] GPTNewNewModel
    - forward

## TFGPTNewNewModel

[[autodoc]] TFGPTNewNewModel
    - call

## FlaxGPTNewNewModel

[[autodoc]] FlaxGPTNewNewModel
    - __call__

"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            doc_file = os.path.join(tmp_dir, "gpt2.mdx")
            new_doc_file = os.path.join(tmp_dir, "gpt-new-new.mdx")

            gpt2_model_patterns = ModelPatterns("GPT2", "gpt2")
            new_model_patterns = ModelPatterns("GPT-New New", "huggingface/gpt-new-new")

            self.init_file(doc_file, test_doc)
            duplicate_doc_file(doc_file, gpt2_model_patterns, new_model_patterns)
            self.check_result(new_doc_file, test_new_doc)

            test_new_doc_pt_only = test_new_doc.replace(
                """
## TFGPTNewNewModel

[[autodoc]] TFGPTNewNewModel
    - call

## FlaxGPTNewNewModel

[[autodoc]] FlaxGPTNewNewModel
    - __call__

""",
                "",
            )
            self.init_file(doc_file, test_doc)
            duplicate_doc_file(doc_file, gpt2_model_patterns, new_model_patterns, frameworks=["pt"])
            self.check_result(new_doc_file, test_new_doc_pt_only)

            test_new_doc_no_tok = test_new_doc.replace(
                """
## GPTNewNewTokenizer

[[autodoc]] GPTNewNewTokenizer
    - save_vocabulary

## GPTNewNewTokenizerFast

[[autodoc]] GPTNewNewTokenizerFast
""",
                "",
            )
            new_model_patterns = ModelPatterns(
                "GPT-New New", "huggingface/gpt-new-new", tokenizer_class="GPT2Tokenizer"
            )
            self.init_file(doc_file, test_doc)
            duplicate_doc_file(doc_file, gpt2_model_patterns, new_model_patterns)
            self.check_result(new_doc_file, test_new_doc_no_tok)

            test_new_doc_pt_only_no_tok = test_new_doc_no_tok.replace(
                """
## TFGPTNewNewModel

[[autodoc]] TFGPTNewNewModel
    - call

## FlaxGPTNewNewModel

[[autodoc]] FlaxGPTNewNewModel
    - __call__

""",
                "",
            )
            self.init_file(doc_file, test_doc)
            duplicate_doc_file(doc_file, gpt2_model_patterns, new_model_patterns, frameworks=["pt"])
            self.check_result(new_doc_file, test_new_doc_pt_only_no_tok)
