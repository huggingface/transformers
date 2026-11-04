# Copyright 2026 HuggingFace Inc.
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
import collections
import copy
import inspect
import os.path
import re
import tempfile
from collections import defaultdict
from typing import get_args

from parameterized import parameterized
from pytest import mark
from safetensors.torch import load_file

from transformers import (
    PreTrainedConfig,
    PreTrainedModel,
    is_torch_available,
    logging,
    set_seed,
)
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import PrefixChange, WeightRenaming, process_target_pattern
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import FLASH_ATTN_KERNEL_FALLBACK, _get_tied_weight_keys
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.testing_utils import (
    CaptureLogger,
    preserve_module_forwards,
    require_flash_attn,
    require_flash_attn_3,
    require_flash_attn_4,
    require_kernels,
    require_torch,
    require_torch_accelerator,
    require_torch_gpu,
    run_first,
    run_test_using_subprocess,
    torch_device,
)
from transformers.utils import (
    SAFE_WEIGHTS_NAME,
    is_torch_bf16_available_on_device,
)
from transformers.utils.output_capturing import OutputRecorder

from ..generation.test_utils import GenerationTesterMixin
from ._helpers import (
    ModelTesterHost,
    _config_supports_rope_scaling,
    _config_zero_init,
    _get_nested_configs,
    _set_config_rope_params,
    compare_state_dicts,
    find_parent_traceback,
    ids_tensor,
    seeded_weight_init,
    skip_weight_init,
)


if is_torch_available():
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file as safe_load_file
    from safetensors.torch import save_file as safe_save_file
    from torch import nn

    from transformers.distributed.parallel_utils import _get_parameter_plan
    from transformers.modeling_utils import load_state_dict
    from transformers.pytorch_utils import id_tensor_storage


@require_torch
class ModelWeightsIOTesterMixin(ModelTesterHost):
    # Models added before 2023 skip `test_can_init_all_missing_weights` unless they set this once their `_init_weights` is fixed
    test_can_init_all_missing_weights_before_2023 = False
    test_missing_keys = True

    def test_from_pretrained_no_checkpoint(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            state_dict = model.state_dict()

            new_model = model_class.from_pretrained(
                pretrained_model_name_or_path=None, config=config, state_dict=state_dict
            )
            new_state_dict = new_model.state_dict()
            assert state_dict.keys() == new_state_dict.keys()
            keys = state_dict.keys()
            for k in keys:
                p1, p2 = new_state_dict[k], state_dict[k]
                with self.subTest(k):
                    torch.testing.assert_close(p1, p2, msg=f"failed on {k}")

            new_params = dict(new_model.named_parameters())
            for k, v in list(model.named_parameters()):
                with self.subTest(k):
                    torch.testing.assert_close(v, new_params[k], msg=f"failed on {k}")

    def test_keep_in_fp32_modules_exist(self):
        """Test that both the `_keep_in_fp32` and `_keep_in_fp32_strict` targets match some layers, to avoid any typo"""
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config))
                # Make sure the modules correctly exist if the flag is active
                if len(model._keep_in_fp32_modules) == 0 and len(model._keep_in_fp32_modules_strict) == 0:
                    self.skipTest(
                        reason=f"{model_class.__name__} has no _keep_in_fp32_modules nor _keep_in_fp32_modules_strict attribute defined"
                    )

                state_dict_names = {k for k, v in model.state_dict().items()}
                # Check that every module in the keep_in_fp32 list is part of the module graph
                if len(model._keep_in_fp32_modules) > 0:
                    non_existent = []
                    for module in model._keep_in_fp32_modules:
                        if not any(re.search(rf"(?:^|\.){module}(?:\.|$)", name) for name in state_dict_names):
                            non_existent.append(module)
                    self.assertTrue(
                        len(non_existent) == 0,
                        f"{non_existent} were specified in the `_keep_in_fp32_modules` list, but are not part of the modules in"
                        f" {model_class.__name__}",
                    )

                if len(model._keep_in_fp32_modules_strict) > 0:
                    non_existent = []
                    for module in model._keep_in_fp32_modules_strict:
                        if not any(re.search(rf"(?:^|\.){module}(?:\.|$)", name) for name in state_dict_names):
                            non_existent.append(module)
                    self.assertTrue(
                        len(non_existent) == 0,
                        f"{non_existent} were specified in the `_keep_in_fp32_modules_strict` list, but are not part of the "
                        f"modules in {model_class.__name__}",
                    )

    def test_keep_in_fp32_modules(self):
        """Test that the flag `_keep_in_fp32_modules` and `_keep_in_fp32_modules_strict`  is correctly respected."""
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config))
                if len(model._keep_in_fp32_modules) == 0 and len(model._keep_in_fp32_modules_strict) == 0:
                    self.skipTest(
                        reason=f"{model_class.__name__} class has no _keep_in_fp32_modules or _keep_in_fp32_modules_strict attribute defined"
                    )

                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)

                    model = model_class.from_pretrained(tmpdirname, dtype=torch.float16)
                    self.assertFalse(
                        model._keep_in_fp32_modules & model._keep_in_fp32_modules_strict,
                        "We found a layer in both the `_keep_in_fp32_modules` and `_keep_in_fp32_modules_strict` lists. Please remove it from one of the two lists.",
                    )
                    # When reloading in fp16, keep_in_fp32_modules AND keep_in_fp32_modules_strict should be upcasted
                    all_fp32_modules = model._keep_in_fp32_modules | model._keep_in_fp32_modules_strict
                    # integer buffers (e.g. step counters) are never cast
                    for name, param in model.state_dict().items():
                        if not param.is_floating_point():
                            continue
                        if any(re.search(rf"(?:^|\.){k}(?:\.|$)", name) for k in all_fp32_modules):
                            self.assertTrue(param.dtype == torch.float32, f"{name} not upcasted to fp32")
                        else:
                            self.assertTrue(param.dtype == torch.float16, f"{name} was upcasted but it should NOT be")

                    # When reloading in bf16, only keep_in_fp32_modules_strict should be upcasted
                    model = model_class.from_pretrained(tmpdirname, dtype=torch.bfloat16)
                    for name, param in model.state_dict().items():
                        if not param.is_floating_point():
                            continue
                        if any(re.search(rf"(?:^|\.){k}(?:\.|$)", name) for k in model._keep_in_fp32_modules_strict):
                            self.assertTrue(param.dtype == torch.float32, f"{name} not upcasted to fp32")
                        else:
                            self.assertTrue(param.dtype == torch.bfloat16, f"{name} was upcasted but it should NOT be")

    def test_load_contiguous_weights(self):
        """
        Checks whether the loaded weights are contiguous or not; inherently checking whether a conversion
        operation from `core_model_loading` may have affected the original weights.
        """
        for model_class in self.all_model_classes:
            config = self.model_tester.get_config()

            model = model_class(config)
            self.assertTrue(all(param.is_contiguous() for param in list(model.parameters())))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                model = model_class.from_pretrained(tmpdirname)
                self.assertTrue(all(param.is_contiguous() for param in list(model.parameters())))

    def test_can_init_all_missing_weights(self):
        """Ensure that all weights are correctly taken into account in `_init_weights`"""
        config = self.model_tester.get_config()

        # This is used to get the addition year of the model
        filename = inspect.getfile(config.__class__)
        # No easy way to get model addition date -> check copyright year on top of file
        with open(filename, encoding="utf-8") as file:
            source_code = file.read()
        addition_year = 0  # if we cannot find it, set it to 0 (i.e. oldest)
        if match_object := re.search(r"^# Copyright (\d{4})", source_code, re.MULTILINE | re.IGNORECASE):
            addition_year = int(match_object.group(1))
        # For now, skip everything older than 2023 and "important models" (too many models to patch otherwise)
        # TODO: relax this as we patch more and more models
        if addition_year < 2023 and not self.test_can_init_all_missing_weights_before_2023:
            self.skipTest(reason="Not a prioritized model for now.")

        for model_class in self.all_model_classes:
            # This context manager makes sure that we get the same results deterministically for random new weights
            with seeded_weight_init():
                # First, initialize the model from __init__ -> this ensure everything is correctly initialized, even if
                # _init_weights() does not take all weights into account correctly
                model_from_init = model_class(copy.deepcopy(config))
                # Here, passing an empty state dict will force all weights to be moved from meta to cpu, then be initialized
                # by _init_weights()
                model_from_pretrained = model_class.from_pretrained(None, config=copy.deepcopy(config), state_dict={})

            # First, check if any parameters/buffers are still on meta -> this is usually an issue with tied weights
            params_on_meta = []
            for k, v in model_from_pretrained.named_parameters():
                if v.device.type == "meta":
                    params_on_meta.append(k)
            for k, v in model_from_pretrained.named_buffers():
                if v.device.type == "meta":
                    params_on_meta.append(k)

            self.assertTrue(
                len(params_on_meta) == 0,
                f"The following keys are still on the meta device, it probably comes from an issue in the tied weights or buffers:\n{params_on_meta}",
            )

            from_pretrained_state_dict = model_from_pretrained.state_dict()
            from_init_state_dict = model_from_init.state_dict()
            self.assertEqual(
                sorted(from_pretrained_state_dict.keys()),
                sorted(from_init_state_dict.keys()),
                "The keys from each model should be the exact same",
            )

            # Everything must be exactly the same as we set the same seed for each init
            different_weights = set()
            for k1, v1 in from_init_state_dict.items():
                # In case using torch.nn.utils.parametrizations on a module, we should skip the resulting keys
                if re.search(r"\.parametrizations\..*?\.original[01]", k1):
                    continue
                v2 = from_pretrained_state_dict[k1]
                # Since we added the seed, they should be exactly the same (i.e. using allclose maybe be wrong due
                # to very low std in init function)
                if not (v1 == v2).all():
                    different_weights.add(k1)

            # Find the parent structure of the weights/buffers that are different for explicit error messages
            unique_bad_module_traceback = set()
            for weight in different_weights.copy():
                weight_name, immediate_parent_class, pretrained_parent_class = find_parent_traceback(
                    weight, model_from_init
                )

                # We cannot control timm model weights initialization, so skip in this case
                if (pretrained_parent_class == "TimmWrapperPreTrainedModel" and "timm_model." in weight) or (
                    pretrained_parent_class == "TimmBackbone" and "_backbone." in weight
                ):
                    different_weights.discard(weight)
                    continue

                # Add it to the traceback
                traceback = (
                    f"`{weight_name}` in module `{immediate_parent_class}` called from `{pretrained_parent_class}`\n"
                )
                unique_bad_module_traceback.add(traceback)

            self.assertTrue(
                len(different_weights) == 0,
                f"The following weights are not properly handled in `_init_weights()` (the model should be able to reinitialize "
                f"them correctly if the model is on meta device)::\n{unique_bad_module_traceback}",
            )

    def test_init_weights_can_init_buffers(self):
        """Ensure that all buffers (persistent and non-persistent) are correctly taken into account in `_init_weights`"""
        config = self.model_tester.get_config()

        # Usually, buffers are not initialized randomly (it's kind of the point of having a Buffer instead of a Parameter...)
        # However, some PositionalEmbedding modules have a `positional_embedding` buffer, initialized randomly with normal
        # distribution and std `config.scale` - set it at 0 here to avoid randomness
        if hasattr(config, "scale"):
            config.scale = 0
        for sub_key in config.sub_configs:
            subconfig = getattr(config, sub_key)
            if subconfig is not None and hasattr(subconfig, "scale"):
                subconfig.scale = 0

        for model_class in self.all_model_classes:
            # First, initialize the model directly with `__init__`, with the context manager making sure that we do
            # not run `initialiaze_weights()`, i.e. buffers are the same as in the modules's `__init__` initial definition
            with skip_weight_init():
                model_from_init = model_class(copy.deepcopy(config))
            # Second, initialize the model fully on meta device, then move everything to cpu and run `init_weights`
            with torch.device("meta"):
                model_from_meta_init = model_class(copy.deepcopy(config))
            # move everything randomly to cpu
            model_from_meta_init.to_empty(device="cpu")
            # Now, run all the inits
            model_from_meta_init.init_weights()

            buffers_from_init = dict(model_from_init.named_buffers())
            buffers_from_meta_init = dict(model_from_meta_init.named_buffers())

            self.assertEqual(
                sorted(buffers_from_init.keys()),
                sorted(buffers_from_meta_init.keys()),
                "The name of the buffers from each model should be the exact same",
            )

            # Buffers are not random usually, so everything must match exactly
            different_buffers = set()
            for k1, v1 in buffers_from_init.items():
                v2 = buffers_from_meta_init[k1]
                if not (v1 == v2).all():
                    different_buffers.add(k1)

            # Find the parent structure of the buffers that are different for explicit error messages
            unique_bad_module_traceback = set()
            for buffer in different_buffers.copy():
                buf_name, immediate_parent_class, pretrained_parent_class = find_parent_traceback(
                    buffer, model_from_init
                )
                # Add it to the traceback
                traceback = (
                    f"`{buf_name}` in module `{immediate_parent_class}` called from `{pretrained_parent_class}`\n"
                )
                unique_bad_module_traceback.add(traceback)

            unique_bad_module_traceback = "".join(unique_bad_module_traceback)
            self.assertTrue(
                len(different_buffers) == 0,
                f"The following buffers are not properly handled in `_init_weights()` (the model should be able to reinitialize "
                f"them correctly if the model is on meta device):\n{unique_bad_module_traceback}",
            )

    def test_torch_save_load(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__), tempfile.TemporaryDirectory() as tmp_dir:
                state_dict = model_class(copy.deepcopy(config)).state_dict()
                path = os.path.join(tmp_dir, "pytorch_model.bin")
                for use_zipfile in (True, False):
                    torch.save(state_dict, path, _use_new_zipfile_serialization=use_zipfile)
                    self.assertTrue(compare_state_dicts(state_dict, load_state_dict(path)))

    def test_save_pretrained_roundtrip(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            with self.subTest(model_class.__name__), tempfile.TemporaryDirectory() as tmp_dir:
                model = model_class(copy.deepcopy(config))
                model.save_pretrained(tmp_dir)

                # config file (and generation config, if the model can generate) are saved
                self.assertTrue(os.path.exists(os.path.join(tmp_dir, "config.json")))
                self.assertEqual(model.can_generate(), os.path.exists(os.path.join(tmp_dir, "generation_config.json")))

                # `_keys_to_ignore_on_save` are in the state dict but not in the file
                state_dict_saved = safe_load_file(os.path.join(tmp_dir, SAFE_WEIGHTS_NAME))
                keys_to_ignore = set(getattr(model, "_keys_to_ignore_on_save", None) or [])
                for k in keys_to_ignore:
                    self.assertIn(k, model.state_dict(), "\n".join(model.state_dict().keys()))
                    self.assertNotIn(k, state_dict_saved, "\n".join(state_dict_saved.keys()))

                # the saved state dict can be loaded back (needed by the Trainer checkpointing API)
                if keys_to_ignore:
                    load_result = model.load_state_dict(state_dict_saved, strict=False)
                    allowed_missing = keys_to_ignore | set(getattr(model, "_tied_weights_keys", None) or [])
                    self.assertTrue(
                        not load_result.missing_keys or set(load_result.missing_keys) == allowed_missing,
                        msg=f"Missing keys: {load_result.missing_keys}\nKeys to ignore: {allowed_missing}",
                    )
                    self.assertEqual(load_result.unexpected_keys, [])

                # Save and load a second time: `from_pretrained` adds config fields that `model(config)` does not,
                # and those fields must survive another save/load
                reloaded = model_class.from_pretrained(tmp_dir)
                reloaded.save_pretrained(tmp_dir)
                reloaded_twice = model_class.from_pretrained(tmp_dir)
                self.assertTrue(compare_state_dicts(reloaded.state_dict(), reloaded_twice.state_dict()))

    def test_correct_missing_keys(self):
        if not self.test_missing_keys:
            self.skipTest(reason="test_missing_keys is set to `False`")

        for model_class in self.all_model_classes:
            config = self.model_tester.get_config()
            model = model_class(config)
            base_model_prefix = model.base_model_prefix

            if hasattr(model, base_model_prefix):
                extra_params = {k: v for k, v in model.named_parameters() if not k.startswith(base_model_prefix)}
                extra_params.update({k: v for k, v in model.named_buffers() if not k.startswith(base_model_prefix)})
                # Some models define this as None
                if model._keys_to_ignore_on_load_missing:
                    for key in model._keys_to_ignore_on_load_missing:
                        extra_params.pop(key, None)

                if not extra_params:
                    # In that case, we *are* on a head model, but every single key is not actual parameters
                    continue

                with tempfile.TemporaryDirectory() as temp_dir_name:
                    model.base_model.save_pretrained(temp_dir_name)
                    model, loading_info = model_class.from_pretrained(temp_dir_name, output_loading_info=True)
                    self.assertGreater(len(loading_info["missing_keys"]), 0, model.__class__.__name__)

    def test_base_model_prefix_loads_base_weights(self):
        """
        `base_model_prefix` has to be the attribute a head stores the base model under: the loader strips or prepends
        it to bridge base and head checkpoints, so any other value silently leaves every base weight random.
        """
        config = self.model_tester.get_config()
        base_model_names = MODEL_MAPPING_NAMES.get(config.model_type, ())
        base_model_names = (base_model_names,) if isinstance(base_model_names, str) else base_model_names
        for model_class in self.all_model_classes:
            model = model_class(config)
            for name, child in model.named_children():
                if type(child).__name__ not in base_model_names:
                    continue
                self.assertEqual(model.base_model_prefix, name)
                base_model, child_state_dict = type(child)(config), child.state_dict()
                # Heads may build their base model differently (no pooler, a mask token...): check the shared weights
                expected_keys = {
                    f"{name}.{key}"
                    for key, tensor in base_model.state_dict().items()
                    if key in child_state_dict and child_state_dict[key].shape == tensor.shape
                }
                self.assertTrue(expected_keys)
                with tempfile.TemporaryDirectory() as tmp:
                    base_model.save_pretrained(tmp)
                    _, info = model_class.from_pretrained(tmp, output_loading_info=True, ignore_mismatched_sizes=True)
                self.assertTrue(info["missing_keys"].isdisjoint(expected_keys), info["missing_keys"] & expected_keys)

    def test_can_use_safetensors(self):
        for model_class in self.all_model_classes:
            config = self.model_tester.get_config()
            model_tied = model_class(config)
            with tempfile.TemporaryDirectory() as d:
                try:
                    model_tied.save_pretrained(d)
                except Exception as e:
                    raise Exception(f"Class {model_class.__name__} cannot be saved using safetensors: {e}")
                with self.subTest(model_class):
                    model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                    # Checking the state dicts are correct
                    reloaded_state = model_reloaded.state_dict()
                    for k, v in model_tied.state_dict().items():
                        with self.subTest(f"{model_class.__name__}.{k}"):
                            torch.testing.assert_close(
                                v,
                                reloaded_state[k],
                                msg=lambda x: (
                                    f"{model_class.__name__}: Tensor {k}: {x}.\n{v}\nvs\n{reloaded_state[k]}\n"
                                    "This probably means that it was not set with the correct value when tying."
                                ),
                            )

                    # Checking the tensor sharing are correct on the new model (weights are properly tied in both cases)
                    ptrs = defaultdict(list)
                    for k, v in model_tied.state_dict().items():
                        ptrs[v.data_ptr()].append(k)

                    shared_ptrs = {k: v for k, v in ptrs.items() if len(v) > 1}

                    for shared_names in shared_ptrs.values():
                        reloaded_ptrs = {reloaded_state[k].data_ptr() for k in shared_names}
                        self.assertEqual(
                            len(reloaded_ptrs),
                            1,
                            f"The shared pointers are incorrect, found different pointers for keys {shared_names}. `__init__` and `from_pretrained` end up not tying the weights the same way.",
                        )

                    # Checking there was no complain of missing weights
                    self.assertEqual(
                        infos["missing_keys"],
                        set(),
                        "These keys were removed when serializing, and were not properly loaded by `from_pretrained`.",
                    )

    def test_load_save_without_tied_weights(self):
        for model_class in self.all_model_classes:
            config = self.model_tester.get_config()
            config.tie_word_embeddings = False

            model = model_class(config)  # we init the model without tie
            # if this test fails later on, it means init tied the weights
            with tempfile.TemporaryDirectory() as d:
                model.save_pretrained(d)
                with safe_open(f"{d}/model.safetensors", framework="pt") as f:
                    serialized_keys = f.keys()

                    model_reloaded, infos = model_class.from_pretrained(d, output_loading_info=True)
                    # Checking the state dicts are correct

                    reloaded_state = model_reloaded.state_dict()
                    for k, v in model.state_dict().items():
                        with self.subTest(k):
                            torch.testing.assert_close(
                                v,
                                reloaded_state[k],
                                msg=lambda x: (
                                    f"{model_class.__name__}: Tensor {k}: {x}. Key {k} was serialized: {k in serialized_keys}. If `False`, this means it was probably aliased and safetensors removed it. If `True` it means `_init_weights` overwrote that key"
                                ),
                            )

                # Checking there was no complain of missing weights
                self.assertEqual(
                    infos["missing_keys"],
                    set(),
                    "Given that the loaded weights are the same, the issue is in `tie_weights`: it tied these keys and removed them from serialization. But because of tiying (hardcoded or not) the previous check is fine.\
                        This can happen if `save_pretrained` remove the targets and not the keys from serialiazation, or you hardcoded `self.xxx = yyy` thus forcing to always tie -> they are removed from serialization.",
                )

    def test_tied_weights_keys(self):
        original_config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            copied_config = copy.deepcopy(original_config)
            copied_config.get_text_config().tie_word_embeddings = True
            copied_config.tie_word_embeddings = True
            model_tied = model_class(copied_config)

            tied_weight_keys = _get_tied_weight_keys(model_tied)
            # If we don't find any tied weights keys, and by default we don't tie the embeddings, it's because the model
            # does not tie them or does not have embedding layer (non-text model)
            if len(tied_weight_keys) == 0 and not getattr(original_config, "tie_word_embeddings", None):
                continue

            ptrs = collections.defaultdict(list)
            for name, tensor in model_tied.state_dict().items():
                ptrs[id_tensor_storage(tensor)].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]

            # Detect we get a hit for each key
            for key in tied_weight_keys:
                is_tied_key = any(re.search(key, p) for group in tied_params for p in group)
                self.assertTrue(
                    is_tied_key,
                    f"{key} is not a tied weight key pattern for {model_class}: {is_tied_key}. With same params: {tied_params}",
                )

            # Removed tied weights found from tied params -> there should only be one left after
            for key in tied_weight_keys:
                for i in range(len(tied_params)):
                    tied_params[i] = [p for p in tied_params[i] if re.search(key, p) is None]

            tied_params = [group for group in tied_params if len(group) > 1]
            self.assertListEqual(
                tied_params,
                [],
                f"Missing `_tied_weights_keys` for {model_class}: add all of {tied_params} except one.",
            )

    def test_model_weights_reload_no_missing_tied_weights(self):
        for model_class in self.all_model_classes:
            config = self.model_tester.get_config()
            model = model_class(config)
            with tempfile.TemporaryDirectory() as tmp_dir:
                model.save_pretrained(tmp_dir)

                # We are nuking ALL weights on file, so every parameter should
                # yell on load. We're going to detect if we yell too much, or too little.
                placeholder_dict = {"tensor": torch.tensor([1, 2])}
                safe_save_file(placeholder_dict, os.path.join(tmp_dir, "model.safetensors"), metadata={"format": "pt"})
                model_reloaded, infos = model_class.from_pretrained(tmp_dir, output_loading_info=True)

                params = dict(model_reloaded.named_parameters())
                params.update(dict(model_reloaded.named_buffers()))
                param_names = set(params.keys())

                missing_keys = set(infos["missing_keys"])

                extra_missing = missing_keys - param_names
                # IMPORTANT Remove tied weights from extra missing: they are normally not warned as missing if their tied
                # counterpart is present but here there are no weights at all so we do get the warning.
                ptrs = collections.defaultdict(list)
                for name, tensor in model_reloaded.state_dict().items():
                    ptrs[id_tensor_storage(tensor)].append(name)
                tied_params = [names for _, names in ptrs.items() if len(names) > 1]
                for group in tied_params:
                    # We remove the group from extra_missing if not all weights from group are in it
                    if len(set(group) - extra_missing) > 0:
                        extra_missing = extra_missing - set(group)

                self.assertEqual(
                    extra_missing,
                    set(),
                    f"This model {model_class.__name__} might be missing some `keys_to_ignore`: {extra_missing}. "
                    f"For debugging, tied parameters are {tied_params}",
                )

                missed_missing = param_names - missing_keys
                # Remove nonpersistent buffers from missed_missing
                buffers = [n for n, _ in model_reloaded.named_buffers()]
                nonpersistent_buffers = {n for n in buffers if n not in model_reloaded.state_dict()}
                missed_missing = missed_missing - nonpersistent_buffers

                if model_reloaded._keys_to_ignore_on_load_missing is None:
                    expected_missing = set()
                else:
                    expected_missing = set()
                    for pattern in model_reloaded._keys_to_ignore_on_load_missing:
                        expected_missing.update({k for k in param_names if re.search(pattern, k) is not None})
                self.assertEqual(
                    missed_missing,
                    expected_missing,
                    f"This model {model_class.__name__} ignores keys {missed_missing} but they look like real"
                    " parameters. If they are non persistent buffers make sure to instantiate them with"
                    " `persistent=False`",
                )

    def test_can_load_ignoring_mismatched_shapes(self):
        set_seed(42)  # deterministic weight init
        configs_no_init = _config_zero_init(self.model_tester.get_config())
        configs_no_init.num_labels = 3

        classification_mappings = [
            MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
            MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
            MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
            MODEL_FOR_VIDEO_CLASSIFICATION_MAPPING_NAMES,
        ]

        for model_class in self.all_model_classes:
            if not any(model_class.__name__ in get_values(m) for m in classification_mappings):
                continue

            with self.subTest(msg=f"Testing {model_class}"), tempfile.TemporaryDirectory() as tmp_dir:
                model = model_class(configs_no_init)
                model.save_pretrained(tmp_dir)

                # Fails when we don't set ignore_mismatched_sizes=True
                with self.assertRaises(RuntimeError):
                    model_class.from_pretrained(tmp_dir, num_labels=42)

                logger = logging.get_logger("transformers.modeling_utils")
                with CaptureLogger(logger) as cl:
                    new_model = model_class.from_pretrained(tmp_dir, num_labels=42, ignore_mismatched_sizes=True)
                self.assertIn("Reinit due to size mismatch", cl.out)

                # Find the name of the module with the mismatched size
                top_linear_modules = [(n, m) for n, m in new_model.named_children() if isinstance(m, nn.Linear)]
                # Some old models have the Linear classification layer inside a ClassificationHead or nn.Sequential
                if len(top_linear_modules) == 0:
                    head = next(
                        (
                            (n, m)
                            for n, m in new_model.named_children()
                            if m.__class__.__name__.endswith("ClassificationHead") or isinstance(m, nn.Sequential)
                        ),
                        None,
                    )
                    if head is None:
                        # Unknown (only xlm, perceiver, levit, flaubert, audio_spectrogram_transformer as of 23/09/2025)
                        continue
                    head_name, head_module = head
                    top_linear_modules = [
                        (f"{head_name}.{n}", m) for n, m in head_module.named_children() if isinstance(m, nn.Linear)
                    ]

                # Usually only 1, but swiftformer and deit have 2 Linear layers using `num_labels`
                mismatched_modules = [n for n, m in top_linear_modules if m.out_features == 42]
                old, new = dict(model.named_parameters()), dict(new_model.named_parameters())
                assert not set(old) - set(new)
                for k, v2 in new.items():
                    v1 = old[k]
                    if not any(k.startswith(m) for m in mismatched_modules):
                        # Every param except the mismatched ones must be identical
                        torch.testing.assert_close(v1, v2, msg=f"{k} does not match: {v1} != {v2}")
                    else:
                        # Linear weights are [out_features, in_features]
                        self.assertEqual(v2.shape[0], 42)
                        # New layer must be ~centered on 0 (not tighter: some models hardcode std=0.02)
                        self.assertLessEqual(v1.data.mean().item(), 1e-1, f"Issue with {k}")

    def test_can_be_initialized_on_meta(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            # If it does not raise here, the test passes
            with torch.device("meta"):
                _ = model_class(copy.deepcopy(config))

    @require_torch_accelerator
    def test_can_load_with_device_context_manager(self):
        config = self.model_tester.get_config()
        # Need to specify index 0 here, as `torch_device` is simply the str of the type, e.g. "cuda"
        device = torch.device(torch_device, index=0)
        for model_class in self.all_model_classes:
            # Need to deepcopy here as it is modified in-place in save_pretrained (it sets sdpa for default attn, which
            # is not supported for e.g. dpt_hybrid)
            model = model_class(copy.deepcopy(config))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                with device:
                    new_model = model_class.from_pretrained(tmpdirname)
                unique_devices = {param.device for param in new_model.parameters()} | {
                    buffer.device for buffer in new_model.buffers()
                }

            self.assertEqual(
                unique_devices, {device}, f"All parameters should be on {device}, but found {unique_devices}."
            )

    # Here we need to run with a subprocess as otherwise setting back the default device to the default value ("cpu")
    # may bring unwanted consequences on other tests. See PR #37553
    @run_first
    @run_test_using_subprocess
    @require_torch_accelerator
    def test_can_load_with_global_device_set(self):
        config = self.model_tester.get_config()
        # Need to specify index 0 here, as `torch_device` is simply the str of the type, e.g. "cuda"
        device = torch.device(torch_device, index=0)
        default_device = torch.get_default_device()
        for model_class in self.all_model_classes:
            # Need to deepcopy here as it is modified in-place in save_pretrained (it sets sdpa for default attn, which
            # is not supported for e.g. dpt_hybrid)
            model = model_class(copy.deepcopy(config))

            # set a global gpu device
            torch.set_default_device(device)

            try:
                with tempfile.TemporaryDirectory() as tmpdirname:
                    model.save_pretrained(tmpdirname)

                    new_model = model_class.from_pretrained(tmpdirname)
                    unique_devices = {param.device for param in new_model.parameters()} | {
                        buffer.device for buffer in new_model.buffers()
                    }
            finally:
                # set back the correct device
                torch.set_default_device(default_device)

            self.assertEqual(
                unique_devices, {device}, f"All parameters should be on {device}, but found {unique_devices}."
            )

    def test_cannot_load_with_meta_device_context_manager(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            # Need to deepcopy here as it is modified in-place in save_pretrained (it sets sdpa for default attn, which
            # is not supported for e.g. dpt_hybrid)
            model = model_class(copy.deepcopy(config))

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                with torch.device("meta"):
                    with self.assertRaisesRegex(
                        RuntimeError, "You are using `from_pretrained` with a meta device context manager"
                    ):
                        _ = model_class.from_pretrained(tmpdirname)

    def test_reverse_loading_mapping(self, check_keys_were_modified=True, skip_base_model=False):
        """Make sure we can load and save correctly the models having any weight renaming mapping or weight conversion
        mapping.
        Note that this test would be better if we could start from the serialized keys, and check that the model
        keys correspond to the weight converions. However, when instantiating a model, it already has the "target"
        keys (or modified keys after mapping) of the conversion mapping, so we have to do it the other way, i.e.
        reverse the conversion and then check that those converted keys match correctly the conversions.

        However, all the checks performed here should ensure everything is going as it should.

        Args:
            check_keys_were_modified (`bool`, *optional*, defaults to `True`):
                Whether to expect keys being modified or not. In some cases, models do not change keys but
                their weights, e.g. via transpose, memory alignment, etc.
            skip_base_model (`bool`, *optional*, defaults to `False`):
                Sometimes, mappings are only visible when applied to the model with head, and not visible on the
                base model. This allows to skip the check on the base model. See e.g. `llava` mapping where this
                is the case. In practice, the mappings are still coherent and a base model can still be loaded from
                the head model, thanks to the `base_model_prefix` which will remove the prefix automatically.
        """
        config = self.model_tester.get_config()

        #  Some MoE models alternate between a classic MLP and a MoE layer, in which case we want to have at
        # lest one MoE layer here to check the mapping
        config_to_set = config.get_text_config(decoder=True)
        config_to_set.first_k_dense_replace = 1  # means that the first layer (idx 0) will be MLP, then MoE
        config_to_set.moe_layer_start_index = 1  # same as above but for Ernie 4.5...
        config_to_set.mlp_only_layers = [0]  # same but for qwens
        config_to_set.num_dense_layers = 1  # lfm2_moe

        for model_class in self.all_model_classes:
            if skip_base_model and "For" not in model_class.__name__:
                continue
            # Each individual model is a subtest
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config))
                # Skip if no conversions
                conversions = get_model_conversion_mapping(model, add_legacy=False)
                if len(conversions) == 0:
                    self.skipTest(f"No conversion found for {model_class}")
                # The PrefixChange conersions are only there for BC with hub checkpoints, but cannot be tested
                # for as we skip them automatically if they are not present in loaded checkpoints (we want to
                # mess up the prefixes only if the loaded checkpoints were doing so as well)
                if all(isinstance(conversion, PrefixChange) for conversion in conversions):
                    self.skipTest(f"Only PrefixChange conversions found for {model_class}")

                # Find the model keys, so the targets according to the conversions
                model_keys = list(model.state_dict().keys())

                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Serialize with reverse mapping
                    model.save_pretrained(tmpdirname)
                    state_dict = load_file(os.path.join(tmpdirname, "model.safetensors"))
                    # Get all the serialized keys that we just saved according to the reverse mapping
                    serialized_keys = list(state_dict.keys())

                if check_keys_were_modified:
                    # They should be different, otherwise we did not perform any mapping
                    self.assertNotEqual(sorted(serialized_keys), sorted(model_keys), "No key mapping was performed!")

                # Check that for each conversion entry, we at least map to one key
                for conversion in conversions:
                    # The PrefixChange conersions are only there for BC with hub checkpoints, but cannot be tested
                    # for as we skip them automatically if they are not present in loaded checkpoints (we want to
                    # mess up the prefixes only if the loaded checkpoints were doing so as well)
                    if isinstance(conversion, PrefixChange):
                        continue

                    # Skip if the conversion is scoped to a sub-model, as the conversion is already tested on the sub-model.
                    # Also, it might be the case that the original/weights model did not include the sub-model (e.g. LlavaForConditionalGeneration)
                    if conversion.scope_prefix is not None:
                        continue

                    # Single pass over serialized_keys: the compiled regex already tests all
                    # pattern branches at once, so one call per key is enough.
                    matched_groups: set[str] = set()
                    for key in serialized_keys:
                        if (match := conversion._scoped_match(key)) is not None:
                            matched_groups.add(match[2].lastgroup)  # "g0", "g1", ...

                    for pattern_index, source_pattern in enumerate(conversion.source_patterns):
                        # Some patterns are written for gen-model only and won't be applied on base model
                        if "lm_head" in source_pattern and model_class not in [
                            *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
                            *get_values(MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES),
                        ]:
                            continue

                        # Sometimes the mappings specify keys that are tied, so absent from the saved state dict
                        if isinstance(conversion, WeightRenaming):
                            # We need to revert the target pattern to make it compatible with regex search
                            target_pattern_reversed = conversion.target_patterns[0]
                            captured_group = process_target_pattern(source_pattern)[1]
                            if captured_group:
                                target_pattern_reversed = target_pattern_reversed.replace(r"\1", captured_group)
                            if any(re.search(target_pattern_reversed, k) for k in model.all_tied_weights_keys.keys()):
                                continue

                        self.assertTrue(
                            f"g{pattern_index}" in matched_groups,
                            f"`{source_pattern}` in `{conversion}` did not match any of the source keys. "
                            "This indicates whether that the pattern is not properly written, or that it could not be reversed correctly",
                        )

                # If everything is still good at this point, let's test that we perform the same operations both when
                # reverting ops from `from_pretrained` and from `__init__`
                with tempfile.TemporaryDirectory() as tmpdirname:
                    # The model was instantiated from __init__ before being saved
                    model.save_pretrained(tmpdirname)
                    state_dict_saved_from_init = load_file(os.path.join(tmpdirname, "model.safetensors"))

                    # Now reload it
                    model_reloaded = model_class.from_pretrained(tmpdirname)

                    # Make sure both loaded state_dict are identical
                    self.assertTrue(compare_state_dicts(model_reloaded.state_dict(), model.state_dict()))

                    # The model was instantiated from `from_pretrained` before being saved
                    model_reloaded.save_pretrained(tmpdirname)
                    state_dict_saved_from_pretrained = load_file(os.path.join(tmpdirname, "model.safetensors"))

                    # Make sure both saved state_dict are identical
                    self.assertTrue(compare_state_dicts(state_dict_saved_from_init, state_dict_saved_from_pretrained))

    def test_can_load_from_already_mapped_keys(self):
        """Test that we can correctly reload a model if we chose `save_original_format=False` in `save_pretrained`,
        i.e. we do not reapply weight conversions when reloading if it was saved correctly already.
        """
        config = self.model_tester.get_config()

        for model_class in self.all_model_classes:
            # Each individual model is a subtest
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config))

                # Skip if no conversions
                conversions = get_model_conversion_mapping(model, add_legacy=False)
                if len(conversions) == 0:
                    self.skipTest(f"No conversion found for {model_class}")

                with tempfile.TemporaryDirectory() as tmpdirname:
                    # Serialize without reverting the mapping
                    model.save_pretrained(tmpdirname, save_original_format=False)
                    model_reloaded = model_class.from_pretrained(tmpdirname)
                    # Make sure both saved state_dict are identical
                    self.assertTrue(compare_state_dicts(model.state_dict(), model_reloaded.state_dict()))


@require_torch
class AttnImplementationTesterMixin(ModelTesterHost):
    _is_composite = False

    def test_config_attn_implementation_setter(self):
        config = self.model_tester.get_config()

        def check_attn_implementation_setter(config: PreTrainedConfig, attn_implementation: str):
            if not config._attn_implementation == attn_implementation:
                raise ValueError(
                    f"Unexpected attn_implementation for config {config.__class__.__name__}: "
                    f"{config._attn_implementation} != {attn_implementation}"
                )
            for attribute_value in config.__dict__.values():
                if isinstance(attribute_value, PreTrainedConfig):
                    check_attn_implementation_setter(attribute_value, attn_implementation)

        # Check that attention implementation can be passed with init args
        config_dict = config.to_diff_dict()
        config_dict.pop("_attn_implementation_internal", None)
        config_dict.pop("_attn_implementation", None)
        config_dict["attn_implementation"] = "eager"
        config = type(config)(**config_dict)
        check_attn_implementation_setter(config, "eager")

        # Check that attention implementation can be set to different value
        config._attn_implementation = "sdpa"
        check_attn_implementation_setter(config, "sdpa")

        config._attn_implementation = "eager"
        check_attn_implementation_setter(config, "eager")

    def test_attn_implementation_composite_models(self):
        """
        Tests if composite models can receive a dict object as attn_implementation, where each key should be
        one of the sub-configs from the model's config.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        for model_class in self.all_model_classes:
            if not self._is_composite:
                self.skipTest("Model is not a composite model.")

            config = self.model_tester.get_config()

            # set eager as it will be the one supported in all models
            # we just need to test if passing 'attn_implementation' as a dict fails or not
            attn_implementation_per_subconfig = {"": "eager"}
            for key in config.sub_configs:
                if getattr(config, key) is not None:
                    attn_implementation_per_subconfig[key] = "eager"

            config._attn_implementation = attn_implementation_per_subconfig
            model = model_class(config)
            for key in config.sub_configs:
                if getattr(config, key) is not None:
                    sub_config = getattr(model.config, key)
                    self.assertTrue(sub_config._attn_implementation == "eager")

            for name, submodule in model.named_modules():
                class_name = submodule.__class__.__name__
                if (
                    class_name.endswith("Attention")
                    and getattr(submodule, "config", None)
                    and submodule.config._attn_implementation != "eager"
                ):
                    raise ValueError(
                        f"The eager model should not have SDPA/FA2 attention layers but got `{class_name}.config._attn_implementation={submodule.config._attn_implementation}`"
                    )

            # Set the attention to default `None` but the text config to `eager`
            # The model should load encoders in SDPA but not the text attention
            config._attn_implementation = None
            config.get_text_config(decoder=True)._attn_implementation = "eager"
            model = model_class(config)
            self.assertTrue(model.config.get_text_config(decoder=True)._attn_implementation == "eager")

            # Test that using `dict` attention implementation works with `from_pretrained`
            #  Set all backbones to "eager" because "eager" attention is always available
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                new_model = model.from_pretrained(tmpdirname, attn_implementation=attn_implementation_per_subconfig)
                self.assertTrue(new_model.config._attn_implementation == "eager")
                for submodule in new_model.modules():
                    if (
                        submodule is not new_model
                        and isinstance(submodule, PreTrainedModel)
                        and submodule.config.__class__ != new_model.config.__class__
                    ):
                        self.assertTrue(submodule.config._attn_implementation == "eager")

    def test_sdpa_can_dispatch_non_composite_models(self):
        """
        Tests if non-composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are called "SDPAAttention".
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self.all_model_classes[0]._supports_sdpa or self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config = self.model_tester.get_config()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.eval().to(torch_device)

                self.assertTrue(model_sdpa.config._attn_implementation == "sdpa")

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.eval().to(torch_device)
                self.assertTrue(model_eager.config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if (
                        class_name.endswith("Attention")
                        and getattr(submodule, "config", None)
                        and submodule.config._attn_implementation == "sdpa"
                    ):
                        raise ValueError(
                            f"The eager model should not have SDPA attention layers but got `{class_name}.config._attn_implementation={submodule.config._attn_implementation}`"
                        )

    def test_sdpa_can_dispatch_composite_models(self):
        """
        Tests if composite models dispatch correctly on SDPA/eager when requested so when loading the model.
        This tests only by looking at layer names, as usually SDPA layers are called "SDPAAttention".
        In contrast to the above test, this one checks if the "config._attn_implementation" is a dict after the model
        is loaded, because we manually replicate requested attn implementation on each sub-config when loading.
        See https://github.com/huggingface/transformers/pull/32238 for more info

        The test tries to cover most general cases of composite models, VLMs with vision and text configs. Any model
        that has a different set of sub-configs has to overwrite this test.
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not self._is_composite:
            self.skipTest(f"{self.all_model_classes[0].__name__} does not support SDPA")

        for model_class in self.all_model_classes:
            config = self.model_tester.get_config()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model_sdpa = model_class.from_pretrained(tmpdirname)
                model_sdpa = model_sdpa.base_model

                modality_tower_names = {
                    "visual",
                    "image_tower",
                    "vision_tower",
                    "vision_model",
                    "audio_tower",
                    "audio_model",
                }
                language_model_names = {"language_model", "model", "text_model"}
                modality_tower_name = [
                    name for name in modality_tower_names if getattr(model_sdpa, name, None) is not None
                ]
                modality_tower_name = modality_tower_name[0] if len(modality_tower_name) > 0 else None
                language_model_name = [name for name in language_model_names if hasattr(model_sdpa, name)]
                language_model_name = language_model_name[0] if len(language_model_name) > 0 else None
                if language_model_name is None or modality_tower_name is None:
                    self.skipTest(
                        reason="Model does not have both a non-text modality tower and a language sub-model, "
                        "cannot test composite SDPA dispatch"
                    )
                modality_tower_sdpa = getattr(model_sdpa, modality_tower_name)
                language_model_sdpa = getattr(model_sdpa, language_model_name)
                text_attn = "sdpa" if language_model_sdpa._supports_sdpa else "eager"
                modality_attn = "sdpa" if modality_tower_sdpa._supports_sdpa else "eager"

                # `None` as it is the requested one which will be assigned to each sub-config
                # Sub-model will dispatch to SDPA if it can (checked below that `SDPA` layers are present)
                self.assertTrue(language_model_sdpa.config._attn_implementation == text_attn)
                self.assertTrue(modality_tower_sdpa.config._attn_implementation == modality_attn)

                model_eager = model_class.from_pretrained(tmpdirname, attn_implementation="eager")
                model_eager = model_eager.base_model
                self.assertTrue(getattr(model_eager, language_model_name).config._attn_implementation == "eager")
                self.assertTrue(getattr(model_eager, modality_tower_name).config._attn_implementation == "eager")

                for name, submodule in model_eager.named_modules():
                    class_name = submodule.__class__.__name__
                    if (
                        class_name.endswith("Attention")
                        and getattr(submodule, "config", None)
                        and submodule.config._attn_implementation == "sdpa"
                    ):
                        raise ValueError("The eager model should not have SDPA attention layers")

    def flash_attn_can_dispatch_composite_models(self, attn_implementation: str):
        """
        Tests if composite models can dispatch on flash attention if the sub-models support it.
        The tests is needed as we handle differently composite models and we cannot check them
        with above tests. If any of the sub-models does not support flash attention, we'll raise an error when dispatching
        that particular sub-model. Otherwise we dispatch safely in all sub-models, where "sub-models" are specific
        backbone models (LM/vision/audio/etc)
        """
        if not self.has_attentions:
            self.skipTest(reason="Model architecture does not support attentions")

        if not is_torch_bf16_available_on_device(torch_device):
            self.skipTest(f"bfloat16 not supported on {torch_device} (on the specific device currently used)")

        if not self._is_composite:
            self.skipTest("This model is not a composite model!")

        dtype = torch.bfloat16
        expected_attn_implementations: set[str] = {attn_implementation}
        if attn_implementation in FLASH_ATTN_KERNEL_FALLBACK:
            expected_attn_implementations.add(FLASH_ATTN_KERNEL_FALLBACK[attn_implementation])

        for model_class in self.all_model_classes:
            config = self.model_tester.get_config()
            model = model_class(config)

            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)
                model = model_class.from_pretrained(tmpdirname, dtype=dtype)

                sub_models_supporting_fa = [
                    module._supports_flash_attn
                    for name, module in model.named_modules()
                    if isinstance(module, PreTrainedModel) and name != ""
                ]
                supports_fa_all_modules = (
                    all(sub_models_supporting_fa) if len(sub_models_supporting_fa) > 0 else model._supports_flash_attn
                )
                if not supports_fa_all_modules:
                    with self.assertRaises(ValueError):
                        model_fa = model_class.from_pretrained(
                            tmpdirname,
                            dtype=dtype,
                            attn_implementation=attn_implementation,
                        )
                else:
                    model_fa = model_class.from_pretrained(
                        tmpdirname, dtype=dtype, attn_implementation=attn_implementation
                    )
                    for key in model_fa.config:
                        if isinstance(getattr(model_fa.config, key), PreTrainedConfig):
                            sub_config = getattr(model_fa.config, key)
                            self.assertIn(sub_config._attn_implementation, expected_attn_implementations)

                    has_fa = False
                    for name, submodule in model_fa.named_modules():
                        class_name = submodule.__class__.__name__
                        if (
                            "Attention" in class_name
                            and getattr(submodule, "config", None)
                            and submodule.config._attn_implementation in expected_attn_implementations
                        ):
                            has_fa = True
                            break
                    if not has_fa:
                        raise ValueError(f"The {attn_implementation} model should have {attn_implementation} layers")

    @require_flash_attn
    @require_torch_accelerator
    @mark.flash_attn_test
    def test_flash_attn_2_can_dispatch_composite_models(self):
        self.flash_attn_can_dispatch_composite_models(attn_implementation="flash_attention_2")

    @require_flash_attn_3
    @require_torch_gpu
    @mark.flash_attn_3_test
    def test_flash_attn_3_can_dispatch_composite_models(self):
        self.flash_attn_can_dispatch_composite_models(attn_implementation="flash_attention_3")

    @require_flash_attn_4
    @require_torch_gpu
    @mark.flash_attn_4_test
    def test_flash_attn_4_can_dispatch_composite_models(self):
        self.flash_attn_can_dispatch_composite_models(attn_implementation="flash_attention_4")

    def test_can_set_attention_dynamically(self):
        """Tests `set_attn_implementation` with a `str`, which should be applied to the model and all its submodels."""
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            if not model_class._can_set_attn_implementation():
                self.skipTest(reason="This model does not support setting its attention dynamically")

            # Need to deepcopy here to avoid changing the _attn_implementation in-place
            model_config = copy.deepcopy(config)
            # Set eager everywhere (it sets it recursively on subconfigs)
            model_config._attn_implementation = "eager"
            model = model_class(model_config)
            # All the configs, at any depth, and the ones of every submodel, as some hold their own copy of the config
            # that is not a sub-config
            all_configs = _get_nested_configs(model.config) + [
                submodule.config for submodule in model.modules() if isinstance(submodule, PreTrainedModel)
            ]

            def check_attn_implementation(expected):
                for nested_config in all_configs:
                    self.assertTrue(nested_config._attn_implementation == expected)

            # sanity check to make sure everything is correctly eager
            check_attn_implementation("eager")

            if not all(
                submodule._can_set_attn_implementation()
                for submodule in model.modules()
                if isinstance(submodule, PreTrainedModel)
            ):
                self.skipTest(reason="Parts of this model cannot set attention dynamically")
            # Some old models technically should support switching, but don't have the flags active...
            if not all(
                submodule._supports_sdpa for submodule in model.modules() if isinstance(submodule, PreTrainedModel)
            ):
                self.skipTest(reason="Parts of this model don't support sdpa")

            # Now, set it to sdpa
            model.set_attn_implementation("sdpa")

            # Check everything was correctly changed
            check_attn_implementation("sdpa")

            # Check we cannot set it to random values, and it raises an error
            with self.assertRaisesRegex(ValueError, 'Specified `attn_implementation="foo"` is not supported'):
                model.set_attn_implementation("foo")

            # Should still be sdpa everywhere
            check_attn_implementation("sdpa")

            # Setting it again should work as well (nothing should be cached from the previous calls)
            model.set_attn_implementation("eager")
            check_attn_implementation("eager")

    def test_can_set_attention_dynamically_composite_model(self):
        """Tests `set_attn_implementation` with a `dict`, where each value should only be applied to its sub-config tree."""
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            if not model_class._can_set_attn_implementation():
                self.skipTest(reason="This model does not support setting its attention dynamically")
            if not self._is_composite:
                self.skipTest(reason="This model is not composite")

            # Need to deepcopy here to avoid changing the _attn_implementation in-place
            model_config = copy.deepcopy(config)
            # Set eager everywhere (it sets it recursively on subconfigs)
            model_config._attn_implementation = "eager"
            model = model_class(model_config)
            nested_configs = _get_nested_configs(model.config)

            def check_attn_implementation(expected_sdpa_configs):
                # All the configs, at any depth: the ones in `expected_sdpa_configs` should be sdpa, the others eager
                for nested_config in nested_configs:
                    expected = "sdpa" if any(nested_config is c for c in expected_sdpa_configs) else "eager"
                    self.assertTrue(nested_config._attn_implementation == expected)

            # sanity check to make sure everything is correctly eager
            check_attn_implementation([])

            if not all(
                submodule._can_set_attn_implementation()
                for submodule in model.modules()
                if isinstance(submodule, PreTrainedModel)
            ):
                self.skipTest(reason="Parts of this model cannot set attention dynamically")

            # Now, set only top-most to sdpa (should support it if it supports the dynamic switch)
            model.set_attn_implementation({"": "sdpa"})

            # Check only top-most was correctly changed, and not the sub-configs at any depth
            check_attn_implementation([model.config])

            # Set each sub-config to a different value than the top-most one: it should be applied to the whole
            # sub-config tree (including the submodels nested in the corresponding submodel), and nothing else
            model.set_attn_implementation({"": "eager"})
            check_attn_implementation([])
            for subconfig_key in model.config.sub_configs:
                if (subconfig := getattr(model.config, subconfig_key)) is None:
                    continue
                subtree_configs = _get_nested_configs(subconfig)
                subtree_submodels = [
                    submodule
                    for submodule in model.modules()
                    if isinstance(submodule, PreTrainedModel) and any(submodule.config is c for c in subtree_configs)
                ]
                if not all(submodel._supports_sdpa for submodel in subtree_submodels):
                    continue

                model.set_attn_implementation({subconfig_key: "sdpa"})
                check_attn_implementation(subtree_configs)
                # Setting it back must work as well, i.e. a previous call must not prevent the next ones
                model.set_attn_implementation({subconfig_key: "eager"})
                check_attn_implementation([])

                # A nested dict targets the sub-configs of the sub-config, as when initializing a config
                for nested_key in subconfig.sub_configs:
                    if (nested_subconfig := getattr(subconfig, nested_key)) is None:
                        continue
                    model.set_attn_implementation({subconfig_key: {nested_key: "sdpa"}})
                    check_attn_implementation(_get_nested_configs(nested_subconfig))
                    model.set_attn_implementation({subconfig_key: {nested_key: "eager"}})
                    check_attn_implementation([])


@require_torch
class ModelInterfaceTesterMixin(ModelTesterHost):
    def test_bc_torch_dtype(self):
        """
        Test that we can still use `torch_dtype` argument correctly, for BC.
        """
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            if "TimmBackbone" in model_class.__name__:
                self.skipTest("TimmBackbone should not run this test")
            # First check that it works correctly
            model = model_class(copy.deepcopy(config))
            with tempfile.TemporaryDirectory() as tmpdirname:
                model.save_pretrained(tmpdirname)

                # Check that it works for all dtypes
                for dtype in ["float16", "bfloat16", "float32", "auto", torch.float16, torch.bfloat16, torch.float32]:
                    model_torch_dtype = model_class.from_pretrained(tmpdirname, torch_dtype=dtype)
                    model_dtype = model_class.from_pretrained(tmpdirname, dtype=dtype)

                    for (k1, v1), (k2, v2) in zip(
                        model_torch_dtype.named_parameters(), model_dtype.named_parameters()
                    ):
                        with self.subTest(f"{dtype} for {model_class.__name__}.{k1}"):
                            self.assertEqual(k1, k2)
                            self.assertEqual(v1.dtype, v2.dtype)
                            torch.testing.assert_close(v1, v2, msg=f"{k1} and  {k2} do not match: {v1} != {v2}")

    def test_tp_plan_matches_params(self):
        """Make sure that each entry of the tp plan matches at least one param (this avoid typos and/or edge cases
        with regexes)"""
        config = self.model_tester.get_config()
        # If none of the config and subconfigs have a tp_plan, then skip (otherwise we should make sure to respect the plan)
        if config.base_model_tp_plan is None and all(
            getattr(getattr(config, key), "base_model_tp_plan", None) is None for key in config.sub_configs
        ):
            self.skipTest("Model does not have a TP plan.")

        # Some MoE models alternate between a classic MLP and a MoE layer, in which case we want to have each one
        # in order to test the whole tp plan
        config_to_set = config.get_text_config()
        config_to_set.first_k_dense_replace = 1  # means that the first layer (idx 0) will be MLP, then MoE
        config_to_set.moe_layer_start_index = 1  # same as above but for Ernie 4.5...
        config_to_set.mlp_only_layers = [0]  # same but for qwens

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            param_names = {name for name, _ in model.named_parameters()} | {name for name, _ in model.named_buffers()}
            module_names = {name for name, _ in model.named_modules()}
            tp_plan = model.tp_plan
            # Make sure the plan is not empty
            self.assertTrue(
                len(tp_plan) > 0,
                f"No TP-plan found for class {model_class.__name__} even though the associated config has one",
            )
            pattern_usage = {}
            for pattern in tp_plan:
                # Check if this given pattern matches any param or module (the value attributed to the pattern does not matter)
                pattern_usage[pattern] = any(
                    _get_parameter_plan(param, {pattern: ""}, is_weight=True) is not None for param in param_names
                ) or any(
                    _get_parameter_plan(module, {pattern: ""}, is_weight=False) is not None for module in module_names
                )

            unused_entries = {k for k, v in pattern_usage.items() if not v}
            self.assertTrue(
                len(unused_entries) == 0, f"The following entries of the TP-plan are not valid: {unused_entries}"
            )

    def test_tied_head_has_a_tp_plan_entry(self):
        """A tied head IS the embedding's parameter, which a tying config shards `embedding_rowwise`,
        so without an entry of its own `F.linear` gets a plain input and a DTensor weight."""
        config = self.model_tester.get_config()
        if config.base_model_tp_plan is None and all(
            getattr(getattr(config, key), "base_model_tp_plan", None) is None for key in config.sub_configs
        ):
            self.skipTest("Model does not have a TP plan.")

        for model_class in self.all_generative_model_classes:
            model = model_class(copy.deepcopy(config))
            head, embedding = model.get_output_embeddings(), model.get_input_embeddings()
            if (
                not model._tied_weights_keys
                or not isinstance(head, nn.Linear)
                or not isinstance(embedding, nn.Embedding)
            ):
                continue
            head_name = next(name for name, module in model.named_modules() if module is head)
            self.assertIn(
                head_name,
                model.tp_plan,
                f"{model_class.__name__} ties `{head_name}` to its input embedding, so a tying checkpoint shards the "
                "weight it reads and it needs a TP-plan entry of its own (`colwise_gather_output`)",
            )

    def test_model_get_set_embeddings(self):
        config = self.model_tester.get_config()

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            self.assertIsInstance(model.get_input_embeddings(), nn.Embedding)

            new_input_embedding_layer = nn.Embedding(10, 10)
            model.set_input_embeddings(new_input_embedding_layer)
            self.assertEqual(model.get_input_embeddings(), new_input_embedding_layer)

            x = model.get_output_embeddings()
            self.assertTrue(x is None or isinstance(x, nn.Linear))

    def test_model_main_input_name(self):
        for model_class in self.all_model_classes:
            model_signature = inspect.signature(getattr(model_class, "forward"))
            # The main input is the name of the argument after `self`
            observed_main_input_name = list(model_signature.parameters.keys())[1]
            self.assertEqual(model_class.main_input_name, observed_main_input_name)

    def test_model_base_model_prefix(self):
        """
        Normally a generative model is a base model + lm_head on top. If this test
        fails for new model, probably the model has incorrect `base_model_prefix` or
        the you are re-defining base blocks for a generative model.
        There are some models which might not fit this assumption, if the model
        has a special architecture. Feel free to skip the test in that case with
        a reason in description.
        """
        for model_class in self.all_generative_model_classes:
            config = self.model_tester.get_config()
            model = model_class(config)
            self.assertTrue(model.base_model is not model)

    def test_internal_model_config_and_subconfig_are_same(self):
        config = self.model_tester.get_config()
        subconfig_keys = list(config.sub_configs.keys())
        for model_class in self.all_model_classes:
            if len(config.sub_configs) == 0:
                self.skipTest(reason="No subconfigs so the test does not make sense")
            # Need to deepcopy here to avoid changing the _attn_implementation in-place
            model = model_class(copy.deepcopy(config))
            subconfig_keys_seen = set()

            for submodule in model.modules():
                # This is a submodel
                if isinstance(submodule, PreTrainedModel) and submodule.config.__class__ != model.config.__class__:
                    subconfig_from_model_internal = submodule.config
                    matching_sub_configs = []
                    for subconfig_key in subconfig_keys:
                        # Get the subconfig from the model config
                        subconfig_from_model_config = getattr(model.config, subconfig_key)
                        if (
                            subconfig_from_model_config is not None
                            and subconfig_from_model_config.__class__ == subconfig_from_model_internal.__class__
                            and subconfig_key not in subconfig_keys_seen
                        ):
                            # Since some composite models have different submodels parameterized by 2 of the same config
                            # class instances, we need to check against a list of matching classes, and check that at least
                            # 1 is the exact object (instead of checking immediately for similar object)
                            matching_sub_configs.append(subconfig_from_model_config)
                            subconfig_keys_seen.add(subconfig_key)

                    # Both should be exactly the same object, that is when instantiating the submodel when should
                    # absolutely not copy the subconfig
                    if len(matching_sub_configs) > 0:
                        self.assertTrue(
                            any(
                                subconfig_from_model_config is subconfig_from_model_internal
                                for subconfig_from_model_config in matching_sub_configs
                            )
                        )

    def test_moe_models_record_router_logits(self):
        """A model with sparse experts has to record `router_logits` and declare them in the output of its generative
        heads, so that `output_router_logits=True` returns them."""
        modeling_source = inspect.getsource(inspect.getmodule(self.all_model_classes[0]))
        if "@use_experts_implementation" not in modeling_source:
            self.skipTest("This model has no sparse experts.")

        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            recordable_outputs = set().union(
                *(
                    (module._can_record_outputs or {}).keys()
                    for module in model.modules()
                    if isinstance(module, PreTrainedModel)
                )
            )
            self.assertIn("router_logits", recordable_outputs, f"{model_class.__name__} does not record them.")
            if model_class in self.all_generative_model_classes:
                return_type = model_class.forward.__annotations__.get("return")
                output_fields = set().union(
                    *(getattr(t, "__dataclass_fields__", {}).keys() for t in (get_args(return_type) or (return_type,)))
                )
                self.assertIn("router_logits", output_fields, f"{model_class.__name__} does not return them.")

    def test_format_of_can_record_outputs(self):
        """Test that that the attribute `_can_record_outputs` is correctly set for a model. It must either be "None" or
        a dictionnary with output names as keys and a recorder or list of recorders as values. A recorder can be an
        OutputRecorder instance, a module class we want to record outputs of or the name of this module class (a str).
        """
        config = self.model_tester.get_config()

        for model_class in self.all_model_classes:
            # Each individual model is a subtest
            with self.subTest(model_class.__name__):
                model = model_class(copy.deepcopy(config)).to(device=torch_device)
                model.eval()

                recordable_output_dicts = [
                    (module._can_record_outputs or {})
                    for module in model.modules()
                    if isinstance(module, PreTrainedModel)
                ]

                # Check that the values of _can_record_outputs are a correct recorder or a list of them
                for recordable_output_dict in recordable_output_dicts:
                    for key, value in recordable_output_dict.items():
                        # Check format of the keys
                        self.assertIsInstance(key, str, f"_can_record_outputs should have str keys, but got {key = }")
                        # Check format of the values
                        recorders = value if isinstance(value, list) else [value]
                        for recorder in recorders:
                            is_valid_recorder = isinstance(recorder, (str, type, OutputRecorder))
                            self.assertTrue(is_valid_recorder, f"Invalid recorder: {recorder}")

    @require_kernels
    @require_torch_accelerator
    def test_kernels_can_load_without_crashing(self):
        """Check whether activating kernels leads to an (value) error"""
        config = self.model_tester.get_config()

        for model_class in self.all_model_classes:
            model = model_class(config).to(torch_device)

            # `kernelize` mutates module-level singletons, so restore them to keep later tests kernel-free
            with preserve_module_forwards(model):
                # Using kernels should not raise a `ValueError`
                model.use_kernels = True

    def test_gradient_checkpointing_backward_compatibility(self):
        config = self.model_tester.get_config()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            config.gradient_checkpointing = True
            model = model_class(copy.deepcopy(config))
            self.assertTrue(model.is_gradient_checkpointing)

    def test_gradient_checkpointing_enable_disable(self):
        config = self.model_tester.get_config()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            # at init model should have gradient checkpointing disabled
            model = model_class(copy.deepcopy(config))
            self.assertFalse(model.is_gradient_checkpointing)

            # Gradient checkpointing is implemented via GradientCheckpointingLayer, if none is present this is likely
            # an implementation issue. Note we exclude clvp for now since they are still not using
            # GradientCheckpointingLayer.
            if config.model_type not in ["clvp", "clvp_decoder"]:
                self.assertTrue([m for m in model.modules() if isinstance(m, GradientCheckpointingLayer)])

            # check enable works
            model.gradient_checkpointing_enable()
            self.assertTrue(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to True
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertTrue(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to True"
                    )

            # check disable works
            model.gradient_checkpointing_disable()
            self.assertFalse(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to False
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertFalse(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to False"
                    )

    def test_peft_gradient_checkpointing_enable_disable(self):
        config = self.model_tester.get_config()

        for model_class in self.all_model_classes:
            if not model_class.supports_gradient_checkpointing:
                continue

            # at init model should have gradient checkpointing disabled
            model = model_class(copy.deepcopy(config))
            self.assertFalse(model.is_gradient_checkpointing)

            # check enable works
            model._hf_peft_config_loaded = True
            try:
                model.gradient_checkpointing_enable()
            except NotImplementedError:
                continue

            self.assertTrue(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to True
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertTrue(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to True"
                    )

            # check disable works
            model.gradient_checkpointing_disable()
            self.assertFalse(model.is_gradient_checkpointing)

            # Loop over all modules and check that relevant modules have gradient_checkpointing set to False
            for n, m in model.named_modules():
                if hasattr(m, "gradient_checkpointing"):
                    self.assertFalse(
                        m.gradient_checkpointing, f"Module {n} does not have gradient_checkpointing set to False"
                    )

    def test_enable_input_require_grads(self):
        config = self.model_tester.get_config()
        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            if not hasattr(model, "get_input_embeddings"):
                continue
            try:
                model.enable_input_require_grads()
            except NotImplementedError as error:
                self.fail(f"enable_input_require_grads raised NotImplementedError for {model_class.__name__}: {error}")
            finally:
                model.disable_input_require_grads()


@require_torch
class ModelRotaryEmbeddingTesterMixin(ModelTesterHost):
    @parameterized.expand([("linear",), ("dynamic",), ("yarn",)])
    def test_model_rope_scaling_from_config(self, scaling_type):
        """
        Tests that we can initialize a model with RoPE scaling in the config, that it can run a forward pass, and
        that a few basic model output properties are honored.
        Note that we test only text backbone's rope module - if vision/audio backbone has RoPE then it will NOT be tested.
        """
        config = self.model_tester.get_config()
        text_config = config.get_text_config(decoder=True)
        base_model_class = None
        for model_class in self.all_model_classes:
            if model_class.__name__ in [
                *get_values(MODEL_MAPPING_NAMES),
            ]:
                base_model_class = model_class
                break

        if base_model_class is None:
            self.skipTest("This model has no `base_model_class` defined in tester.")

        if not _config_supports_rope_scaling(text_config):
            self.skipTest("This model does not support RoPE scaling")

        if not hasattr(text_config, "vocab_size"):
            self.skipTest("This model has no vocab size defined and the test doesn't yet support non-text modalities.")

        n_required_args = sum(
            p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.POSITIONAL_ONLY)
            and p.name != "self"
            for p in inspect.signature(base_model_class.forward).parameters.values()
        )
        if n_required_args > 1:
            self.skipTest("This model requires more than single main input, skip for now as it's not supported")

        short_input = ids_tensor([1, 10], text_config.vocab_size)
        long_input = ids_tensor([1, int(text_config.max_position_embeddings * 1.5)], text_config.vocab_size)
        short_model_kwargs = long_model_kwargs = {}

        if config.is_encoder_decoder:
            short_model_kwargs = {"decoder_input_ids": short_input.clone()}
            long_model_kwargs = {"decoder_input_ids": long_input.clone()}

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        _set_config_rope_params(
            text_config,
            {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "original_max_position_embeddings": 16384,
            },
        )
        original_model = base_model_class(config)
        original_model.to(torch_device)
        original_model.eval()
        original_short_output = original_model(short_input, **short_model_kwargs)
        original_long_output = original_model(long_input, **long_model_kwargs)

        original_short_output = original_short_output.last_hidden_state
        original_long_output = original_long_output.last_hidden_state

        set_seed(42)  # Fixed seed at init time so the two models get the same random weights
        _set_config_rope_params(
            text_config,
            {
                "rope_type": scaling_type,
                "factor": 10.0,
                "rope_theta": 10_000.0,
            },
        )
        scaled_model = base_model_class(config)
        scaled_model.to(torch_device)
        scaled_model.eval()
        scaled_short_output = scaled_model(short_input, **short_model_kwargs).last_hidden_state
        scaled_long_output = scaled_model(long_input, **long_model_kwargs).last_hidden_state

        # Dynamic scaling does not change the RoPE embeddings until it receives an input longer than the original
        # maximum sequence length, so the outputs for the short input should match.
        if scaling_type == "dynamic":
            torch.testing.assert_close(original_short_output, scaled_short_output, rtol=1e-5, atol=1e-5)
        else:
            self.assertFalse(torch.allclose(original_short_output, scaled_short_output, atol=1e-5))

        # The output should be different for long inputs
        self.assertFalse(torch.allclose(original_long_output, scaled_long_output, atol=1e-5))

    def test_model_rope_scaling_frequencies(self):
        """
        Tests the frequency properties of the different RoPE scaling types on the model RoPE layer.
        Note that we test only text backbone's rope module - if vision/audio backbone has RoPE then it will NOT be tested.
        """
        config = self.model_tester.get_config()
        text_config = config.get_text_config(decoder=True)
        base_model_class = None
        for model_class in self.all_model_classes:
            if model_class.__name__ in [
                *get_values(MODEL_MAPPING_NAMES),
            ]:
                base_model_class = model_class
                break

        if base_model_class is None:
            self.skipTest("This model has no `base_model_class` defined in tester.")

        if not _config_supports_rope_scaling(text_config):
            self.skipTest("This model does not support RoPE scaling")

        # Retrieves the RoPE layer class from the base model class. Uses `.named_modules()` to avoid hardcoding the
        # named location of the RoPE layer class.
        base_model = base_model_class(config)
        possible_rope_attributes = [
            "pos_emb",
            "rotary_emb",  # most common case
        ]
        rope_class = None
        for name, module in base_model.named_modules():
            if (
                any(potential_name in name for potential_name in possible_rope_attributes)
                # skip if module doesn't accept config - old API/model
                and (
                    len(params := list(inspect.signature(module.__init__).parameters.values())) >= 1
                    and params[0].name == "config"
                )
                # FIXME: raushan, vision RoPE layers are not standard and can't be tested here
                # thus skip if modules doesn't operate on config. See https://github.com/huggingface/transformers/issues/46443
                and "Vision" not in module.__class__.__name__
            ):
                rope_class = type(module)
                break

        if rope_class is None:
            self.skipTest("This model has no standardized RoPE module found.")

        is_nested_rope = (
            "rope_theta" not in text_config.rope_parameters.keys()
            and "rope_theta" in list(text_config.rope_parameters.values())[0]
        )

        scaling_factor = 10
        short_input_length = 10
        partial_rotary_factor = text_config.rope_parameters.get(
            "partial_rotary_factor", getattr(text_config, "partial_rotary_factor", 1.0)
        )
        long_input_length = int(text_config.max_position_embeddings * 1.5)

        kwargs = {}
        if is_nested_rope:
            kwargs = {"layer_type": list(text_config.rope_parameters.keys())[0]}

        # Inputs
        x = torch.randn(
            1, dtype=torch.float32, device=torch_device
        )  # used exclusively to get the dtype and the device
        position_ids_short = torch.arange(short_input_length, dtype=torch.long, device=torch_device)
        position_ids_short = position_ids_short.unsqueeze(0)
        position_ids_long = torch.arange(long_input_length, dtype=torch.long, device=torch_device)
        position_ids_long = position_ids_long.unsqueeze(0)

        # Infer number of mrope axis which is usually `3` but can be different in special models
        if getattr(text_config, "layer_types", None) is None or set(text_config.rope_parameters.keys()).isdisjoint(
            text_config.layer_types
        ):
            has_per_layer_rope = False
        else:
            has_per_layer_rope = True

        num_multimodal_rope_axis = None
        if not has_per_layer_rope and "mrope_section" in text_config.rope_parameters:
            num_multimodal_rope_axis = len(text_config.rope_parameters["mrope_section"])
        else:
            mrope_sections = [
                sub_dict["mrope_section"]
                for sub_dict in text_config.rope_parameters.values()
                if isinstance(sub_dict, dict) and "mrope_section" in sub_dict
            ]
            if mrope_sections:
                if len({tuple(sections) for sections in mrope_sections}) > 1:
                    raise ValueError(
                        "Model has different `mrope_section` per layer type, override the test if needed!"
                    )
                num_multimodal_rope_axis = len(mrope_sections[0])

        if num_multimodal_rope_axis is not None:
            position_ids_short = position_ids_short[None, ...].repeat(num_multimodal_rope_axis, 1, 1)
            position_ids_long = position_ids_long[None, ...].repeat(num_multimodal_rope_axis, 1, 1)

        # Sanity check original RoPE
        _set_config_rope_params(
            text_config,
            {"rope_type": "default", "rope_theta": 10_000.0, "partial_rotary_factor": partial_rotary_factor},
        )
        original_rope = rope_class(config=text_config).to(torch_device)
        original_cos_short, original_sin_short = original_rope(x, position_ids_short, **kwargs)
        original_cos_long, original_sin_long = original_rope(x, position_ids_long, **kwargs)
        torch.testing.assert_close(original_cos_short, original_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(original_sin_short, original_sin_long[:, :short_input_length, :])

        # Sanity check linear RoPE scaling
        # New position "x" should match original position with index "x/scaling_factor"
        _set_config_rope_params(
            text_config,
            {
                "rope_type": "linear",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        linear_scaling_rope = rope_class(config=text_config).to(torch_device)
        linear_cos_short, linear_sin_short = linear_scaling_rope(x, position_ids_short, **kwargs)
        linear_cos_long, linear_sin_long = linear_scaling_rope(x, position_ids_long, **kwargs)
        torch.testing.assert_close(linear_cos_short, linear_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(linear_sin_short, linear_sin_long[:, :short_input_length, :])
        for new_position in range(0, long_input_length, scaling_factor):
            original_position = int(new_position // scaling_factor)
            torch.testing.assert_close(linear_cos_long[:, new_position, :], original_cos_long[:, original_position, :])
            torch.testing.assert_close(linear_sin_long[:, new_position, :], original_sin_long[:, original_position, :])

        # Sanity check Dynamic NTK RoPE scaling
        # Scaling should only be observed after a long input is fed. We can observe that the frequencies increase
        # with scaling_factor (or that `inv_freq` decreases)
        _set_config_rope_params(
            text_config,
            {
                "rope_type": "dynamic",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        ntk_scaling_rope = rope_class(config=text_config).to(torch_device)
        ntk_cos_short, ntk_sin_short = ntk_scaling_rope(x, position_ids_short, **kwargs)
        ntk_cos_long, ntk_sin_long = ntk_scaling_rope(x, position_ids_long, **kwargs)
        torch.testing.assert_close(ntk_cos_short, original_cos_short)
        torch.testing.assert_close(ntk_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(ntk_sin_long, original_sin_long)
        # CHeck each layer type for nested RoPE configs.
        # Allow one float32 eps of slack, as some devices round `pow` differently
        slack = 1 + torch.finfo(torch.float32).eps
        if not is_nested_rope:
            self.assertTrue((ntk_scaling_rope.inv_freq <= original_rope.inv_freq * slack).all())
        else:
            layer_types = getattr(text_config, "_rope_type_labels", getattr(text_config, "layer_types"))
            for layer_type in layer_types:
                if text_config.rope_parameters[layer_type] is not None:
                    self.assertTrue(
                        (
                            getattr(ntk_scaling_rope, f"{layer_type}_inv_freq")
                            <= getattr(original_rope, f"{layer_type}_inv_freq") * slack
                        ).all()
                    )

        # Sanity check Yarn RoPE scaling
        # Scaling should be over the entire input
        _set_config_rope_params(
            text_config,
            {
                "rope_type": "yarn",
                "factor": scaling_factor,
                "rope_theta": 10_000.0,
                "partial_rotary_factor": partial_rotary_factor,
            },
        )
        yarn_scaling_rope = rope_class(config=text_config).to(torch_device)
        yarn_cos_short, yarn_sin_short = yarn_scaling_rope(x, position_ids_short, **kwargs)
        yarn_cos_long, yarn_sin_long = yarn_scaling_rope(x, position_ids_long, **kwargs)
        torch.testing.assert_close(yarn_cos_short, yarn_cos_long[:, :short_input_length, :])
        torch.testing.assert_close(yarn_sin_short, yarn_sin_long[:, :short_input_length, :])
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_short, original_cos_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_short, original_sin_short)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_cos_long, original_cos_long)
        with self.assertRaises(AssertionError):
            torch.testing.assert_close(yarn_sin_long, original_sin_long)

    def test_vision_axial_rope(self):
        config = self.model_tester.get_config()

        base_model_class = None
        for model_class in self.all_model_classes:
            if model_class.__name__ in [
                *get_values(MODEL_MAPPING_NAMES),
            ]:
                base_model_class = model_class
                break

        if base_model_class is None:
            self.skipTest("This model has no `base_model_class` defined in tester.")

        rope_class = None
        base_model = base_model_class(config)
        for name, module in base_model.named_modules():
            if hasattr(module, "compute_axial_rope_parameters"):
                rope_class = type(module)
                vision_config = module.config
                break

        if rope_class is None:
            self.skipTest(f"{base_model_class} has no axial RoPE layer defined.")

        # First make sure that validation on default config raises no rope-related warnings
        logger = logging.get_logger("transformers.modeling_rope_utils")
        with CaptureLogger(logger) as cl:
            vision_config.validate_rope()
        self.assertEqual("", cl.out)
        logger.warning_once.cache_clear()

        # Axial rope type expects only `rope_theta`, otherwise raises warning
        vision_config.rope_parameters["factor"] = 0.25
        logger = logging.get_logger("transformers.modeling_rope_utils")
        with CaptureLogger(logger) as cl:
            vision_config.validate_rope()
        self.assertEqual("Unrecognized keys in `rope_parameters` for 'rope_type'='axial': {'factor'}\n", cl.out)
        del vision_config.rope_parameters["factor"]
        logger.warning_once.cache_clear()

        inv_freq, attention_scale = rope_class.compute_axial_rope_parameters(config=vision_config)
        rope_module = rope_class(vision_config).to(device=torch_device)

        self.assertTrue(hasattr(rope_module, "inv_freq"))
        self.assertTrue(hasattr(rope_module, "attention_scaling"))
        self.assertEqual(attention_scale, 1.0)  # attention scale is always 1
        torch.testing.assert_close(inv_freq, rope_module.inv_freq.cpu())

        # create 2D position IDs for a single grid of one row and 10 cols `size=(10, 2)`
        position_ids = torch.stack(
            [
                torch.arange(10, dtype=torch.long, device=torch_device),
                torch.zeros(10, dtype=torch.long, device=torch_device),
            ]
        ).transpose(0, 1)
        # and an empty hidden states used only to infer device/dtype
        hidden_states = torch.empty(1, dtype=torch.float32, device=torch_device)
        cos, sin = rope_module(hidden_states, position_ids)
        self.assertEqual(cos.shape[-1], inv_freq.shape[-1] * 4)  # the freq are `//4` of head dim

    def test_model_rope_with_partial_rotation(self):
        config = self.model_tester.get_config()
        text_config = config.get_text_config(decoder=True)
        base_model_class = None
        for model_class in self.all_model_classes:
            if model_class.__name__ in [*get_values(MODEL_MAPPING_NAMES)]:
                base_model_class = model_class
                break

        if base_model_class is None:
            self.skipTest("This model has no `base_model_class` defined in tester.")

        if not hasattr(text_config, "rope_parameters"):
            self.skipTest("This model does not have RoPE")

        if not hasattr(text_config, "vocab_size"):
            self.skipTest("This model has no vocab size defined and the test doesn't yet support non-text modalities.")

        n_required_args = sum(
            p.default is inspect.Parameter.empty
            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY, p.POSITIONAL_ONLY)
            and p.name != "self"
            for p in inspect.signature(base_model_class.forward).parameters.values()
        )
        if n_required_args > 1:
            self.skipTest("This model requires more than single main input, skip for now as it's not supported")

        input_ids = ids_tensor([1, 10], text_config.vocab_size)
        model_kwargs = {}
        input_dict = {}  # FIXME raushan
        if base_model_class.main_input_name != "input_ids":
            model_kwargs[base_model_class.main_input_name] = input_dict[base_model_class.main_input_name][:1]
        else:
            model_kwargs = {"input_ids": input_ids}

        if config.is_encoder_decoder:
            model_kwargs["decoder_input_ids"] = input_ids.clone()

        if "partial_rotary_factor" not in text_config.rope_parameters:
            self.skipTest("This model does not have partial rope supported")

        # Run with partial rotary factor set to a values less than one, should not raise any shape errors
        # If tested already has a value > 1, use it since that might affect to other config field values
        default_partial_rotation = text_config.rope_parameters["partial_rotary_factor"]
        _set_config_rope_params(
            text_config,
            {
                "rope_type": "default",
                "rope_theta": 10_000.0,
                "partial_rotary_factor": 0.5 if default_partial_rotation >= 1.0 else default_partial_rotation,
            },
        )
        model = base_model_class(config)
        model.to(torch_device).eval()
        model(**model_kwargs)


class TesterInterfaceMixin(ModelTesterHost):
    def test_num_layers_is_small(self):
        # TODO (if possible): Avoid exceptional cases, especially for `OwlViT`.
        # ⛔ DO NOT edit this list (unless there is really nothing to tweak in the model tester class and approved by the reviewer) ⛔!
        exceptional_num_hidden_layers = {
            # TODO: There might be some way to fix
            "FunnelModelTest": 5,
            "FunnelBaseModelTest": 4,
            "GroupViTVisionModelTest": 12,
            "OwlViTModelTest": 12,
            "OwlViTTextModelTest": 12,
            "OwlViTForObjectDetectionTest": 12,
            "Owlv2ModelTest": 12,
            "Owlv2TextModelTest": 12,
            "Owlv2ForObjectDetectionTest": 12,
            "Qwen2_5OmniThinkerForConditionalGenerationModelTest": 4,
            "Qwen3OmniMoeThinkerForConditionalGenerationModelTest": 4,
            "SamHQModelTest": 12,
            "Swin2SRModelTest": 3,
            "XLNetModelTest": 3,
            "DPTModelTest": 4,  # `test_modeling_dpt_hybrid.py`: not able to get it work after change `num_hidden_layers` and `neck_hidden_sizes`
            # Nothing we can't do
            "Gemma3nTextModelTest": 4,  # need to test KV shared layer for both types: `full_attention` and `sliding_attention`
            "Gemma3nVision2TextModelTest": 4,  # need to test KV shared layer for both types: `full_attention` and `sliding_attention`
            "MistralLarge4ModelTest": 4,  # need to test KV shared layer for both types: RoPE and NoPE
            "BeitModelTest": 4,  # BeitForSemanticSegmentation requires config.out_indices to be a list of 4 integers
            "ZambaModelTest": 5,  # The minimum number to test beyond the initial ["mamba", "mamba", "hybrid"] in `ZambaConfig._layers_block_type`
            "Wav2Vec2BertModelTest": 4,
        }
        target_num_hidden_layers = exceptional_num_hidden_layers.get(type(self).__name__, 2)

        if hasattr(self.model_tester, "num_hidden_layers") and isinstance(self.model_tester.num_hidden_layers, int):
            assert self.model_tester.num_hidden_layers <= target_num_hidden_layers

        if hasattr(self.model_tester, "vision_config") and "num_hidden_layers" in self.model_tester.vision_config:
            if isinstance(self.model_tester.vision_config, dict):
                assert self.model_tester.vision_config["num_hidden_layers"] <= target_num_hidden_layers
            else:
                assert self.model_tester.vision_config.num_hidden_layers <= target_num_hidden_layers
        if hasattr(self.model_tester, "text_config") and "num_hidden_layers" in self.model_tester.text_config:
            if isinstance(self.model_tester.text_config, dict):
                assert self.model_tester.text_config["num_hidden_layers"] <= target_num_hidden_layers
            else:
                assert self.model_tester.text_config.num_hidden_layers <= target_num_hidden_layers

    def test_model_is_small(self):
        # Just a consistency check to make sure we are not running tests on 1M parameter models.
        config = self.model_tester.get_config()

        for model_class in self.all_model_classes:
            model = model_class(copy.deepcopy(config))
            num_params = model.num_parameters()
            assert num_params < 1000000, (
                f"{model_class} is too big for the common tests ({num_params})! It should have 1M max."
            )

    def test_generation_tester_mixin_inheritance(self):
        """
        Ensures that we have the generation tester mixin if the model can generate. The test will fail otherwise,
        forcing the mixin to be added -- and ensuring proper test coverage
        """
        if len(self.all_generative_model_classes) > 0:
            self.assertTrue(
                issubclass(self.__class__, GenerationTesterMixin),
                msg=(
                    "This model can call `generate` from `GenerationMixin`, so one of two things must happen: 1) the "
                    "tester must inherit from `GenerationTesterMixin` to run `generate` tests, or 2) if the model "
                    "doesn't fully support the original `generate` or has a custom `generate` with partial feature "
                    "support, the tester must overwrite `all_generative_model_classes` to skip the failing classes "
                    "(make sure to comment why). If `all_generative_model_classes` is overwritten as `()`, then we "
                    "need to remove the `GenerationTesterMixin` inheritance -- no `generate` tests are being run."
                ),
            )
        else:
            self.assertFalse(
                issubclass(self.__class__, GenerationTesterMixin),
                msg=(
                    "This model can't call `generate`, so its tester can't inherit `GenerationTesterMixin`. (If you "
                    "think the model should be able to `generate`, the model may be missing the `GenerationMixin` "
                    "inheritance)"
                ),
            )
