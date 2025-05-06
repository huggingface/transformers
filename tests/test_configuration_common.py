# Copyright 2019 HuggingFace Inc.
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
import json
import os
import tempfile
from pathlib import Path

from transformers import is_torch_available
from transformers.utils import direct_transformers_import

from .utils.test_configuration_utils import config_common_kwargs


transformers_module = direct_transformers_import(Path(__file__).parent)


class ConfigTester:
    def __init__(self, parent, config_class=None, has_text_modality=True, common_properties=None, **kwargs):
        self.parent = parent
        self.config_class = config_class
        self.has_text_modality = has_text_modality
        self.inputs_dict = kwargs
        self.common_properties = common_properties

    def create_and_test_config_common_properties(self):
        config = self.config_class(**self.inputs_dict)
        common_properties = (
            ["hidden_size", "num_attention_heads", "num_hidden_layers"]
            if self.common_properties is None and not self.config_class.sub_configs
            else self.common_properties
        )
        common_properties = [] if common_properties is None else common_properties

        # Add common fields for text models
        if self.has_text_modality:
            common_properties.extend(["vocab_size"])

        # Test that config has the common properties as getters
        for prop in common_properties:
            self.parent.assertTrue(hasattr(config, prop), msg=f"`{prop}` does not exist")

        # Test that config has the common properties as setter
        for idx, name in enumerate(common_properties):
            try:
                setattr(config, name, idx)
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

        # Test if config class can be called with Config(prop_name=..)
        for idx, name in enumerate(common_properties):
            try:
                config = self.config_class(**{name: idx})
                self.parent.assertEqual(
                    getattr(config, name), idx, msg=f"`{name} value {idx} expected, but was {getattr(config, name)}"
                )
            except NotImplementedError:
                # Some models might not be able to implement setters for common_properties
                # In that case, a NotImplementedError is raised
                pass

    def create_and_test_config_to_json_string(self):
        config = self.config_class(**self.inputs_dict)
        obj = json.loads(config.to_json_string())
        for key, value in self.inputs_dict.items():
            self.parent.assertEqual(obj[key], value)

    def create_and_test_config_to_json_file(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            json_file_path = os.path.join(tmpdirname, "config.json")
            config_first.to_json_file(json_file_path)
            config_second = self.config_class.from_json_file(json_file_path)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_from_and_save_pretrained(self):
        config_first = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config_first.save_pretrained(tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

        with self.parent.assertRaises(OSError):
            self.config_class.from_pretrained(f".{tmpdirname}")

    def create_and_test_config_from_and_save_pretrained_subfolder(self):
        config_first = self.config_class(**self.inputs_dict)

        subfolder = "test"
        with tempfile.TemporaryDirectory() as tmpdirname:
            sub_tmpdirname = os.path.join(tmpdirname, subfolder)
            config_first.save_pretrained(sub_tmpdirname)
            config_second = self.config_class.from_pretrained(tmpdirname, subfolder=subfolder)

        self.parent.assertEqual(config_second.to_dict(), config_first.to_dict())

    def create_and_test_config_from_and_save_pretrained_composite(self):
        """
        Tests that composite or nested configs can be loaded and saved correctly. In case the config
        has a sub-config, we should be able to call `sub_config.from_pretrained('general_config_file')`
        and get a result same as if we loaded the whole config and obtained `config.sub_config` from it.
        """
        config = self.config_class(**self.inputs_dict)

        with tempfile.TemporaryDirectory() as tmpdirname:
            config.save_pretrained(tmpdirname)
            general_config_loaded = self.config_class.from_pretrained(tmpdirname)
            general_config_dict = config.to_dict()

            # Iterate over all sub_configs if there are any and load them with their own classes
            sub_configs = general_config_loaded.sub_configs
            for sub_config_key, sub_class in sub_configs.items():
                if sub_class.__name__ == "AutoConfig":
                    sub_class = sub_class.for_model(**general_config_dict[sub_config_key]).__class__
                    sub_config_loaded = sub_class.from_pretrained(tmpdirname)
                else:
                    sub_config_loaded = sub_class.from_pretrained(tmpdirname)

                # Pop `transformers_version`, it never exists when a config is part of a general composite config
                # Verify that loading with subconfig class results in same dict as if we loaded with general composite config class
                sub_config_loaded_dict = sub_config_loaded.to_dict()
                sub_config_loaded_dict.pop("transformers_version", None)
                self.parent.assertEqual(sub_config_loaded_dict, general_config_dict[sub_config_key])

                # Verify that the loaded config type is same as in the general config
                type_from_general_config = type(getattr(general_config_loaded, sub_config_key))
                self.parent.assertTrue(isinstance(sub_config_loaded, type_from_general_config))

                # Now save only the sub-config and load it back to make sure the whole load-save-load pipeline works
                with tempfile.TemporaryDirectory() as tmpdirname2:
                    sub_config_loaded.save_pretrained(tmpdirname2)
                    sub_config_loaded_2 = sub_class.from_pretrained(tmpdirname2)
                    self.parent.assertEqual(sub_config_loaded.to_dict(), sub_config_loaded_2.to_dict())

    def create_and_test_config_with_num_labels(self):
        config = self.config_class(**self.inputs_dict, num_labels=5)
        self.parent.assertEqual(len(config.id2label), 5)
        self.parent.assertEqual(len(config.label2id), 5)

        config.num_labels = 3
        self.parent.assertEqual(len(config.id2label), 3)
        self.parent.assertEqual(len(config.label2id), 3)

    def check_config_can_be_init_without_params(self):
        if self.config_class.has_no_defaults_at_init:
            with self.parent.assertRaises(ValueError):
                config = self.config_class()
        else:
            config = self.config_class()
            self.parent.assertIsNotNone(config)

    def check_config_arguments_init(self):
        if self.config_class.sub_configs:
            return  # TODO: @raushan composite models are not consistent in how they set general params

        kwargs = copy.deepcopy(config_common_kwargs)
        config = self.config_class(**kwargs)
        wrong_values = []
        for key, value in config_common_kwargs.items():
            if key == "torch_dtype":
                if not is_torch_available():
                    continue
                else:
                    import torch

                    if config.torch_dtype != torch.float16:
                        wrong_values.append(("torch_dtype", config.torch_dtype, torch.float16))
            elif getattr(config, key) != value:
                wrong_values.append((key, getattr(config, key), value))

        if len(wrong_values) > 0:
            errors = "\n".join([f"- {v[0]}: got {v[1]} instead of {v[2]}" for v in wrong_values])
            raise ValueError(f"The following keys were not properly set in the config:\n{errors}")

    def run_common_tests(self):
        self.create_and_test_config_common_properties()
        self.create_and_test_config_to_json_string()
        self.create_and_test_config_to_json_file()
        self.create_and_test_config_from_and_save_pretrained()
        self.create_and_test_config_from_and_save_pretrained_subfolder()
        self.create_and_test_config_from_and_save_pretrained_composite()
        self.create_and_test_config_with_num_labels()
        self.check_config_can_be_init_without_params()
        self.check_config_arguments_init()
