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
import inspect
import json
import os
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Literal, Optional

from ..conversion_mapping import get_model_conversion_mapping
from ..core_model_loading import (
    Concatenate,
    ConversionOps,
    MergeModulelist,
    Transpose,
    WeightConverter,
    WeightRenaming,
)
from ..utils import (
    CONFIG_NAME,
    cached_file,
    check_peft_version,
    extract_commit_hash,
    find_adapter_config_file,
    is_accelerate_available,
    is_peft_available,
    is_torch_available,
    logging,
)
from ..utils.hub import DownloadKwargs
from ..utils.loading_report import log_state_dict_report


if is_torch_available():
    import torch

if is_accelerate_available():
    from accelerate import dispatch_model
    from accelerate.utils import get_balanced_memory, infer_auto_device_map

# Minimum PEFT version supported for the integration
MIN_PEFT_VERSION = "0.18.0"


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    from ..modeling_utils import LoadStateDictConfig


def _block_diag_3d(*tensors):
    lora_b_block_diag = []
    for i in range(len(tensors[0])):
        lora_b_block_diag.append(torch.block_diag(tensors[0][i], tensors[1][i]))
    out = torch.stack(lora_b_block_diag, dim=0)
    return out


class PeftConcatenate(Concatenate):
    """Convert per-expert LoRA weights to merged weights.

    When the base weights are fused, e.g. W01 = [W0, W1], the LoRA weights also need to be fused. To achieve this
    correctly, concatenate the LoRA A weights along the r (rank) dimension. This doesn't require a new Operation. But
    for LoRA B, the weights need to be merged in a block diagonal fashion to achieve the correct result.

    To illustrate:

    Before
    W0' = W0 + A0 @ B0
    W1' = W1 + A1 @ B1

    After
    W01' = W01 + A01 @ B01_bd
        where
        A01 = [A0, A1]
        B01_bd = [[B0,  0],
                  [0,  B1]]

    This class is responsible for merging LoRA B in this block-diagonal fashion. Assuming that we fuse N weights, it
    should look like this:

    1. LoRA B is 2-dim
    Normal LoRA weight of shape (out_feat, rank), the output shape should be (N * out_feat, N * rank).

    2. LoRA B is 3-dim
    MoE LoRA weight of shape (experts, out_feat, rank), the output shape should be (experts, N * out_feat, N * rank).

    After this, the experts x rank dimension are flattened, as PEFT expects 2d tensors for LoRA.
    """

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        full_layer_name: str,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        dims = [v.dim() for v in input_dict.values()]
        if set(dims) not in ({2}, {3}):
            raise ValueError(
                f"To convert this LoRA adapter, the LoRA weights all need to have either 2 or 3 dims, got {set(dims)}"
            )

        if set(dims) == {2}:
            output_dict = {full_layer_name: torch.block_diag(*input_dict.values())}
        else:
            out = _block_diag_3d(*input_dict.values())  # shape = experts, 2*out_feat, 2*r
            out = torch.permute(out, (2, 0, 1))  # shape = 2*r, experts, 2*out_feat
            out = out.flatten(0, 1)  # shape = 2*r * experts, 2*out_feat
            out = out.T
            output_dict = {full_layer_name: out}
        return output_dict

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("Reversing PEFT LoRA MoE conversions is not supported yet.")


class FlattenDims(ConversionOps):
    """
    Flatten the tensors along the given dimensions
    """

    def __init__(self, dims: int | tuple[int, ...]):
        if isinstance(dims, int):
            dims = (dims,)
        self.dims = dims

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        output_dict = {k: v.flatten(*self.dims) for k, v in input_dict.items()}
        return output_dict

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("Reversing flatteing operatio is not supported.")

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims})"


class PermuteDims(ConversionOps):
    """
    Permute the tensors along the given dimensions
    """

    def __init__(self, dims: tuple[int, ...]):
        self.dims = dims

    @torch.no_grad
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        config,
        **kwargs,
    ) -> dict[str, list[torch.Tensor]]:
        output_dict = {k: v.permute(*self.dims) for k, v in input_dict.items()}
        return output_dict

    @property
    def reverse_op(self) -> ConversionOps:
        raise NotImplementedError("Reversing flatteing operatio is not supported yet.")

    def __repr__(self):
        return f"{self.__class__.__name__}(dims={self.dims})"


def _build_peft_weight_mapping(
    weight_conversions: list[WeightConverter | WeightRenaming] | None, adapter_name: str
) -> list[WeightConverter | WeightRenaming]:
    # We iterate over all the operations of the original model and simply edit them to apply to the PEFT adapter when
    # appropriate.
    if not weight_conversions:
        return []

    # strip "base_model.model" and add adapter name
    new_weight_conversions = [
        WeightRenaming("base_model.model.model", "model"),
        WeightRenaming("lora_A.weight", f"lora_A.{adapter_name}.weight"),
        WeightRenaming("lora_B.weight", f"lora_B.{adapter_name}.weight"),
        # TODO: lora_embedding_A and B
    ]

    for orig_conversion in weight_conversions:
        if isinstance(orig_conversion, WeightRenaming):
            new_weight_conversions.append(orig_conversion)
            continue

        if orig_conversion.target_patterns == ["mlp.experts.gate_up_proj"]:
            # gate_up_proj requires both merging the experts and concatenating for the fusion of w1 and w3
            for lora in ("lora_A", "lora_B"):  # TODO: lora_embedding_A and lora_embedding_B
                conversion = copy.deepcopy(orig_conversion)
                # deal with operations
                peft_weight_operations = []
                for i, op in enumerate(conversion.operations):
                    if isinstance(op, Concatenate):
                        if lora == "lora_B":  # block diagonal concat
                            peft_weight_operations.append(PeftConcatenate(dim=op.dim))
                        else:  # normal concat + flatten
                            peft_weight_operations.append(op)
                            peft_weight_operations.append(FlattenDims(dims=(0, 1)))
                    elif isinstance(op, MergeModulelist):
                        peft_weight_operations.append(op)
                conversion.operations = peft_weight_operations

                # TODO: this assumption may not hold for models != mixtral
                # For source, we capture the orignal weights + the lora weights
                new_source_patterns = []
                for pat in list(conversion.source_patterns):
                    # we replace the weight pattern to colllect loras
                    pat = pat.rsplit(".", 1)[0]
                    # note: the source state_dict does *not* contain the adapter name
                    new_source_patterns.append(f"{pat}.{lora}.*")
                conversion.source_patterns = new_source_patterns

                # the gate_up_proj is the innner PEFT ParamWrapper, so we need to use base_layer
                pat = conversion.target_patterns[0]
                pat = pat.replace("gate_up_proj", "base_layer")
                # we make sure the target key is correct, add '.weight' because the parameter is targeted directly
                conversion.target_patterns = [f"{pat}.{lora}.{adapter_name}.weight"]
                new_weight_conversions.append(conversion)

        elif orig_conversion.target_patterns == ["mlp.experts.down_proj"]:
            # down_proj only requires merging of experts
            for lora in ("lora_A", "lora_B"):  # TODO: lora_embedding_A and lora_embedding_B
                conversion = copy.deepcopy(orig_conversion)
                peft_weight_operations = []
                for i, op in enumerate(conversion.operations):
                    if isinstance(op, MergeModulelist):
                        peft_weight_operations.append(op)
                        if lora == "lora_A":
                            peft_weight_operations.append(FlattenDims(dims=(0, 1)))
                        else:
                            peft_weight_operations.append(PermuteDims(dims=(2, 0, 1)))
                            peft_weight_operations.append(FlattenDims(dims=(0, 1)))
                            peft_weight_operations.append(Transpose(dim0=0, dim1=1))
                conversion.operations = peft_weight_operations

                # TODO: this assumption may not hold for models != mixtral
                # For source, we capture the orignal weights + the lora weights
                new_source_patterns = []
                for pat in list(conversion.source_patterns):
                    # we replace the weight pattern to colllect loras
                    pat = pat.rsplit(".", 1)[0]
                    # note: the source state_dict does *not* contain the adapter name
                    new_source_patterns.append(f"{pat}.{lora}.*")
                conversion.source_patterns = new_source_patterns

                # the down_proj is the outer PEFT ParamWrapper, so we remove the prefix
                pat = conversion.target_patterns[0]
                pat = pat.replace(".down_proj", "")
                # we make sure the target key is correct, add '.weight' because the parameter is targeted directly
                conversion.target_patterns = [f"{pat}.{lora}.{adapter_name}.weight"]
                new_weight_conversions.append(conversion)

    return new_weight_conversions


def patch_mixtral_moe_parameter_targeting(model, peft_config):
    """PEFT currently assumes that expert layers are of shape
        (expert, in, out)
    but with Mixtral in transformers v5 this is not true anymore.
    This will be addressed in PEFT >0.19 until then we need to handle
    it here for now.
    """
    import peft
    from functools import wraps

    if model.config.model_type == "mixtral":
        get_in_out_features = peft.tuners.lora.layer.ParamWrapper._get_in_out_features

        @wraps(get_in_out_features)
        def new_get_in_out_features(layer, module):
            if layer.parameter_name in ("down_proj", "gate_up_proj"):
                in_features, out_features = get_in_out_features(layer, module)
                return out_features, in_features
            return get_in_out_features(layer, module)

        peft.tuners.lora.layer.ParamWrapper._get_in_out_features = new_get_in_out_features


class PeftAdapterMixin:
    """
    A class containing all functions for loading and using adapters weights that are supported in PEFT library. For
    more details about adapters and injecting them on a transformer-based model, check out the documentation of PEFT
    library: https://huggingface.co/docs/peft/index

    Currently supported PEFT methods are all non-prompt learning methods (LoRA, IA³, etc.). Other PEFT models such as
    prompt tuning, prompt learning are out of scope as these adapters are not "injectable" into a torch module. For
    using these methods, please refer to the usage guide of PEFT library.

    With this mixin, if the correct PEFT version is installed (>= 0.18.0), it is possible to:

    - Load an adapter stored on a local path or in a remote Hub repository, and inject it in the model
    - Attach new adapters in the model and train them with Trainer or by your own.
    - Attach multiple adapters and iteratively activate / deactivate them
    - Activate / deactivate all adapters from the model.
    - Get the `state_dict` of the active adapter.
    """

    _hf_peft_config_loaded = False
    _prepare_peft_hotswap_kwargs: dict | None = None

    def load_adapter(
        self,
        peft_model_id: str | None = None,
        adapter_name: str | None = None,
        peft_config: dict[str, Any] | None = None,
        adapter_state_dict: dict[str, "torch.Tensor"] | None = None,
        low_cpu_mem_usage: bool = False,
        is_trainable: bool = False,
        hotswap: bool | Literal["auto"] = "auto",
        local_files_only: bool = False,
        adapter_kwargs: dict[str, Any] | None = None,
        load_config: Optional["LoadStateDictConfig"] = None,
        **kwargs,
    ) -> None:
        """
        Load adapter weights from file or remote Hub folder. If you are not familiar with adapters and PEFT methods, we
        invite you to read more about them on PEFT official documentation: https://huggingface.co/docs/peft

        Requires PEFT to be installed as a backend to load the adapter weights.

        Args:
            peft_model_id (`str`, *optional*):
                The identifier of the model to look for on the Hub, or a local path to the saved adapter config file
                and adapter weights.
            adapter_name (`str`, *optional*):
                The adapter name to use. If not set, will use the name "default".
            load_config (`LoadStateDictConfig`, *optional*):
                A load configuration to reuse when pulling adapter weights, typically from `from_pretrained`.
            kwargs (`dict[str, Any]`, *optional*):
                Additional `LoadStateDictConfig` fields passed as keyword arguments.
            peft_config (`dict[str, Any]`, *optional*):
                The configuration of the adapter to add, supported adapters are all non-prompt learning configs (LoRA,
                IA³, etc). This argument is used in case users directly pass PEFT state dicts.
            adapter_state_dict (`dict[str, torch.Tensor]`, *optional*):
                The state dict of the adapter to load. This argument is used in case users directly pass PEFT state
                dicts.
            low_cpu_mem_usage (`bool`, *optional*, defaults to `False`):
                Reduce memory usage while loading the PEFT adapter. This should also speed up the loading process.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and can only be
                used for inference.
            hotswap : (`"auto"` or `bool`, *optional*, defaults to `"auto"`)
                Whether to substitute an existing (LoRA) adapter with the newly loaded adapter in-place. This means
                that, instead of loading an additional adapter, this will take the existing adapter weights and replace
                them with the weights of the new adapter. This can be faster and more memory efficient. However, the
                main advantage of hotswapping is that when the model is compiled with torch.compile, loading the new
                adapter does not require recompilation of the model. When using hotswapping, the passed `adapter_name`
                should be the name of an already loaded adapter.

                If the new adapter and the old adapter have different ranks and/or LoRA alphas (i.e. scaling), you need
                to call an additional method before loading the adapter:

                ```py
                model = AutoModel.from_pretrained(...)
                max_rank = ...  # the highest rank among all LoRAs that you want to load
                # call *before* compiling and loading the LoRA adapter
                model.enable_peft_hotswap(target_rank=max_rank)
                model.load_adapter(file_name_1, adapter_name="default")
                # optionally compile the model now
                model = torch.compile(model, ...)
                output_1 = model(...)
                # now you can hotswap the 2nd adapter, use the same name as for the 1st
                # hotswap is activated by default since enable_peft_hotswap was called
                model.load_adapter(file_name_2, adapter_name="default")
                output_2 = model(...)
                ```

                By default, hotswap is disabled and requires passing `hotswap=True`. If you called
                `enable_peft_hotswap` first, it is enabled. You can still manually disable it in that case by passing
                `hotswap=False`.

                Note that hotswapping comes with a couple of limitations documented here:
                https://huggingface.co/docs/peft/main/en/package_reference/hotswap
            adapter_kwargs (`dict[str, Any]`, *optional*):
                Additional keyword arguments passed along to the `from_pretrained` method of the adapter config and
                `find_adapter_config_file` method.
        """
        from peft import PeftType

        from ..modeling_utils import LoadStateDictConfig, _get_resolved_checkpoint_files

        if local_files_only:
            kwargs["local_files_only"] = True
        base_load_config = load_config.__dict__ if load_config is not None else {}
        base_load_config.update(kwargs)
        base_load_config.setdefault("pretrained_model_name_or_path", None)
        load_config = LoadStateDictConfig(**base_load_config)
        peft_model_id = peft_model_id or load_config.pretrained_model_name_or_path

        if hotswap == "auto":
            # if user called model.enable_peft_hotswap and this is not the first adapter, enable hotswap
            hotswap_enabled = getattr(self, "_hotswap_enabled", False)
            not_first_adapter = bool(self._hf_peft_config_loaded and (adapter_name in self.peft_config))
            hotswap = hotswap_enabled and not_first_adapter

        if hotswap:
            if (not self._hf_peft_config_loaded) or (adapter_name not in self.peft_config):
                raise ValueError(
                    "To hotswap an adapter, there must already be an existing adapter with the same adapter name."
                )
            if any(conf.peft_type != PeftType.LORA for conf in self.peft_config.values()):
                raise ValueError("Hotswapping is currently only supported for LoRA, please set `hotswap=False`.")

        adapter_name = adapter_name if adapter_name is not None else "default"
        adapter_kwargs = adapter_kwargs or {}

        weight_conversions = get_model_conversion_mapping(self)
        peft_weight_conversions = _build_peft_weight_mapping(weight_conversions, adapter_name)

        from peft import PeftConfig, inject_adapter_in_model

        if self._hf_peft_config_loaded and (not hotswap) and (adapter_name in self.peft_config):
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")
        elif hotswap and ((not self._hf_peft_config_loaded) or (adapter_name not in self.peft_config)):
            raise ValueError(
                "To hotswap an adapter, there must already be an existing adapter with the same adapter name."
            )

        if peft_model_id is None and (adapter_state_dict is None and peft_config is None):
            raise ValueError(
                "You should either pass a `peft_model_id` or a `peft_config` and `adapter_state_dict` to load an adapter."
            )

        if peft_config is None:
            adapter_config_file = find_adapter_config_file(
                peft_model_id,
                **load_config.download_kwargs,
            )

            if adapter_config_file is None:
                raise ValueError(
                    f"adapter model file not found in {peft_model_id}. Make sure you are passing the correct path to the "
                    "adapter model."
                )

            peft_config = PeftConfig.from_pretrained(
                peft_model_id,
                **load_config.download_kwargs,
                **adapter_kwargs,
            )
            peft_config.inference_mode = not is_trainable

        peft_config = convert_peft_config_for_transformers(peft_config, model=self, conversions=weight_conversions)

        patch_mixtral_moe_parameter_targeting(model=self, peft_config=peft_config)

        if not hotswap:
            # Create and add fresh new adapters into the model, unless the weights are hotswapped
            inject_adapter_in_model(peft_config, self, adapter_name)

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True

        if adapter_state_dict is None:
            checkpoint_files, sharded_metadata = _get_resolved_checkpoint_files(
                pretrained_model_name_or_path=peft_model_id,
                variant=None,
                gguf_file=None,
                use_safetensors=load_config.use_safetensors,
                user_agent={},
                is_remote_code=False,
                transformers_explicit_filename="adapter_model.bin"
                if load_config.use_safetensors is False
                else "adapter_model.safetensors",
                download_kwargs=load_config.download_kwargs,
            )
        else:
            checkpoint_files, sharded_metadata = [], {}

        load_config = replace(
            load_config,
            pretrained_model_name_or_path=peft_model_id,
            sharded_metadata=sharded_metadata,
            weight_mapping=peft_weight_conversions,
        )
        load_info = self._load_pretrained_model(
            model=self,
            state_dict=adapter_state_dict,
            checkpoint_files=checkpoint_files,
            load_config=load_config,
        )

        adapter_key_markers = {adapter_name}
        if peft_config is not None and getattr(peft_config, "peft_type", None) is not None:
            adapter_key_markers.add(peft_config.peft_type.value.lower())

        def is_adapter_key(key: str) -> bool:
            return any(marker in key for marker in adapter_key_markers)

        unexpected_keys = load_info.unexpected_keys
        missing_keys = [key for key in load_info.missing_keys if is_adapter_key(key)]
        mismatched_keys = [item for item in load_info.mismatched_keys if is_adapter_key(item[0])]

        log_state_dict_report(
            model=self,
            load_config=load_config,
            logger=logger,
            error_msgs=load_info.error_msgs,
            unexpected_keys=unexpected_keys,
            missing_keys=missing_keys,
            mismatched_keys=mismatched_keys,
            mismatched_shapes=mismatched_keys,
            conversion_errors=load_info.conversion_errors,
        )

    def enable_peft_hotswap(
        self, target_rank: int = 128, check_compiled: Literal["error", "warn", "ignore"] = "error"
    ) -> None:
        """Enables the possibility to hotswap PEFT adapters with different ranks, or, if the model is compiled, without
        triggering recompilation.

        Right now, hotswapping is only supported for LoRA.

        Calling this method is only required when hotswapping adapters and if the model is compiled or if the ranks of
        the loaded adapters differ. If the ranks are all identical and the model is not compiled, hotswapping works
        without calling this method first.

        Args:
            target_rank (`int`, *optional*, defaults to `128`):
                The highest rank among all the adapters that will be loaded.
            check_compiled (`str`, *optional*, defaults to `"error"`):
                How to handle the case when the model is already compiled, which should generally be avoided. The
                options are:
                  - "error" (default): raise an error
                  - "warn": issue a warning
                  - "ignore": do nothing
        """
        if getattr(self, "peft_config", {}):
            if check_compiled == "error":
                raise RuntimeError("Call `enable_peft_hotswap` before loading the first adapter.")
            elif check_compiled == "warn":
                logger.warning(
                    "It is recommended to call `enable_peft_hotswap` before loading the first adapter to avoid recompilation."
                )
            elif check_compiled != "ignore":
                raise ValueError(
                    f"check_compiles should be one of 'error', 'warn', or 'ignore', got '{check_compiled}' instead."
                )

        self._hotswap_enabled = True
        self._prepare_peft_hotswap_kwargs = {"target_rank": target_rank, "check_compiled": check_compiled}

    def add_adapter(self, adapter_config, adapter_name: str | None = None) -> None:
        r"""
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Adds a fresh new adapter to the current model for training purpose. If no adapter name is passed, a default
        name is assigned to the adapter to follow the convention of PEFT library (in PEFT we use "default" as the
        default adapter name).

        Note that the newly added adapter is not automatically activated. To activate it, use `model.set_adapter`.

        Args:
            adapter_config (`~peft.PeftConfig`):
                The configuration of the adapter to add, supported adapters are non-prompt learning methods (LoRA,
                IA³, etc.).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to add. If no name is passed, a default name is assigned to the adapter.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        from peft import PeftConfig, inject_adapter_in_model

        adapter_name = adapter_name or "default"

        if not self._hf_peft_config_loaded:
            self._hf_peft_config_loaded = True
        elif adapter_name in self.peft_config:
            raise ValueError(f"Adapter with name {adapter_name} already exists. Please use a different name.")

        if not isinstance(adapter_config, PeftConfig):
            raise TypeError(f"adapter_config should be an instance of PeftConfig. Got {type(adapter_config)} instead.")

        # Retrieve the name or path of the model, one could also use self.config._name_or_path
        # but to be consistent with what we do in PEFT: https://github.com/huggingface/peft/blob/6e783780ca9df3a623992cc4d1d665001232eae0/src/peft/mapping.py#L100
        adapter_config.base_model_name_or_path = self.__dict__.get("name_or_path", None)
        # TODO: WE NEED TOO APPLY OUR DYNAMIC WEIGHT CONVERSION AT SOME POINT HERE!
        inject_adapter_in_model(adapter_config, self, adapter_name)

        self.set_adapter(adapter_name)

    def set_adapter(self, adapter_name: list[str] | str) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Sets a specific adapter by forcing the model to use a that adapter and disable the other adapters.

        Args:
            adapter_name (`Union[list[str], str]`):
                The name of the adapter to set. Can be also a list of strings to set multiple adapters.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)
        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")
        elif isinstance(adapter_name, list):
            missing = set(adapter_name) - set(self.peft_config)
            if len(missing) > 0:
                raise ValueError(
                    f"Following adapter(s) could not be found: {', '.join(missing)}. Make sure you are passing the correct adapter name(s)."
                    f" current loaded adapters are: {list(self.peft_config.keys())}"
                )
        elif adapter_name not in self.peft_config:
            raise ValueError(
                f"Adapter with name {adapter_name} not found. Please pass the correct adapter name among {list(self.peft_config.keys())}"
            )

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        _adapters_has_been_set = False

        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.set_adapter(adapter_name)
                _adapters_has_been_set = True

        if not _adapters_has_been_set:
            raise ValueError(
                "Did not succeeded in setting the adapter. Please make sure you are using a model that supports adapters."
            )

    def disable_adapters(self) -> None:
        r"""
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Disable all adapters that are attached to the model. This leads to inferring with the base model only.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer
        from peft.utils import ModulesToSaveWrapper

        for _, module in self.named_modules():
            if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                module.enable_adapters(enabled=False)

    def enable_adapters(self) -> None:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Enable adapters that are attached to the model.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                module.enable_adapters(enabled=True)

    def active_adapters(self) -> list[str]:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the current active adapters of the model. In case of multi-adapter inference (combining multiple adapters
        for inference) returns the list of all active adapters so that users can deal with them accordingly.

        For previous PEFT versions (that does not support multi-adapter inference), `module.active_adapter` will return
        a single string.
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.tuners.tuners_utils import BaseTunerLayer

        for _, module in self.named_modules():
            if isinstance(module, BaseTunerLayer):
                active_adapters = module.active_adapter
                break

        # For previous PEFT versions
        if isinstance(active_adapters, str):
            active_adapters = [active_adapters]

        return active_adapters

    def get_adapter_state_dict(self, adapter_name: str | None = None, state_dict: dict | None = None) -> dict:
        """
        If you are not familiar with adapters and PEFT methods, we invite you to read more about them on the PEFT
        official documentation: https://huggingface.co/docs/peft

        Gets the adapter state dict that should only contain the weights tensors of the specified adapter_name adapter.
        If no adapter_name is passed, the active adapter is used.

        Args:
            adapter_name (`str`, *optional*):
                The name of the adapter to get the state dict from. If no name is passed, the active adapter is used.
            state_dict (nested dictionary of `torch.Tensor`, *optional*)
                The state dictionary of the model. Will default to `self.state_dict()`, but can be used if special
                precautions need to be taken when recovering the state dictionary of a model (like when using model
                parallelism).
        """
        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft import get_peft_model_state_dict

        if adapter_name is None:
            adapter_name = self.active_adapters()[0]

        adapter_state_dict = get_peft_model_state_dict(self, state_dict=state_dict, adapter_name=adapter_name)
        return adapter_state_dict

    def _dispatch_accelerate_model(
        self,
        device_map: str,
        max_memory: int | None = None,
        offload_folder: str | None = None,
        offload_index: int | None = None,
    ) -> None:
        """
        Optional re-dispatch the model and attach new hooks to the model in case the model has been loaded with
        accelerate (i.e. with `device_map=xxx`)

        Args:
            device_map (`str` or `dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.

                To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For
                more information about each option see [designing a device
                map](https://hf.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map).
            max_memory (`Dict`, *optional*):
                A dictionary device identifier to maximum memory. Will default to the maximum memory available for each
                GPU and the available CPU RAM if unset.
            offload_folder (`str` or `os.PathLike`, *optional*):
                If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
            offload_index (`int`, *optional*):
                The offload_index argument to be passed to `accelerate.dispatch_model` method.
        """
        dispatch_model_kwargs = {}
        # Safety checker for previous `accelerate` versions
        # `offload_index` was introduced in https://github.com/huggingface/accelerate/pull/873/
        if "offload_index" in inspect.signature(dispatch_model).parameters:
            dispatch_model_kwargs["offload_index"] = offload_index

        no_split_module_classes = self._no_split_modules

        if device_map != "sequential":
            max_memory = get_balanced_memory(
                self,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                low_zero=(device_map == "balanced_low_0"),
            )
        if isinstance(device_map, str):
            device_map = infer_auto_device_map(
                self, max_memory=max_memory, no_split_module_classes=no_split_module_classes
            )
        dispatch_model(
            self,
            device_map=device_map,
            offload_dir=offload_folder,
            **dispatch_model_kwargs,
        )

    def delete_adapter(self, adapter_names: list[str] | str) -> None:
        """
        Delete a PEFT adapter from the underlying model.

        Args:
            adapter_names (`Union[list[str], str]`):
                The name(s) of the adapter(s) to delete.
        """

        check_peft_version(min_version=MIN_PEFT_VERSION)

        if not self._hf_peft_config_loaded:
            raise ValueError("No adapter loaded. Please load an adapter first.")

        from peft.functional import delete_adapter

        if isinstance(adapter_names, str):
            adapter_names = [adapter_names]

        # Check that all adapter names are present in the config
        missing_adapters = [name for name in adapter_names if name not in self.peft_config]
        if missing_adapters:
            raise ValueError(
                f"The following adapter(s) are not present and cannot be deleted: {', '.join(missing_adapters)}"
            )

        prefixes = [f"{self.peft_config[adapter_name].peft_type.value.lower()}_" for adapter_name in adapter_names]
        for adapter_name, prefix in zip(adapter_names, prefixes):
            delete_adapter(self, adapter_name=adapter_name, prefix=prefix)
            # For transformers integration - we need to pop the adapter from the config
            if getattr(self, "_hf_peft_config_loaded", False) and hasattr(self, "peft_config"):
                self.peft_config.pop(adapter_name, None)

        # In case all adapters are deleted, we need to delete the config
        # and make sure to set the flag to False
        if len(self.peft_config) == 0:
            del self.peft_config
            self._hf_peft_config_loaded = False


def maybe_load_adapters(
    pretrained_model_name_or_path,
    download_kwargs: DownloadKwargs,
    **adapter_kwargs,
):
    if pretrained_model_name_or_path is None or not is_peft_available():
        return None, pretrained_model_name_or_path, adapter_kwargs

    token = download_kwargs.get("token")

    if download_kwargs.get("commit_hash") is None:
        resolved_config_file = cached_file(
            pretrained_model_name_or_path,
            CONFIG_NAME,
            cache_dir=download_kwargs.get("cache_dir"),
            force_download=bool(download_kwargs.get("force_download", False)),
            proxies=download_kwargs.get("proxies"),
            local_files_only=bool(download_kwargs.get("local_files_only", False)),
            token=token,
            revision=download_kwargs.get("revision"),
            subfolder=download_kwargs.get("subfolder"),
            _raise_exceptions_for_gated_repo=False,
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
        )
        download_kwargs["commit_hash"] = extract_commit_hash(resolved_config_file, None)

    _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

    token_from_adapter_kwargs = adapter_kwargs.pop("token", None)

    if _adapter_model_path is None:
        peft_kwargs = adapter_kwargs.copy()
        for arg_name in ("cache_dir", "proxies", "subfolder"):  # don't override revision
            if (arg_name not in peft_kwargs) and (arg_name in download_kwargs):
                peft_kwargs[arg_name] = download_kwargs[arg_name]
        if "commit_hash" in download_kwargs:
            peft_kwargs["_commit_hash"] = download_kwargs["commit_hash"]
        peft_kwargs["force_download"] = bool(download_kwargs.get("force_download", False))
        peft_kwargs["local_files_only"] = bool(download_kwargs.get("local_files_only", False))
        peft_kwargs["token"] = token or token_from_adapter_kwargs
        _adapter_model_path = find_adapter_config_file(
            pretrained_model_name_or_path,
            **peft_kwargs,
        )

    if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
        with open(_adapter_model_path, "r", encoding="utf-8") as f:
            _adapter_model_path = pretrained_model_name_or_path
            pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]

    return _adapter_model_path, pretrained_model_name_or_path, adapter_kwargs


#####################
# weight conversion #
#####################

# With transformers v5, we need to convert some weights to reflect updated model architectures. If users have trained
# PEFT adapters for these models, they also need to be updated. This may require updating the PEFT config too. The
# logic for this is found below. Right now, only LoRA is supported.

# TODO: These functions will be added to PEFT in release 0.19.0. Drop them here once 0.19.0 becomes the min PEFT
# version.


def _convert_peft_config_mixtral(peft_config):
    peft_config.target_parameters = peft_config.target_parameters or set()

    # add gate.weight to target_parameters
    for target in peft_config.target_modules:
        if (target == "gate") or target.endswith(".gate"):
            # FIXME: what if only specific layers are targeted, e.g. '0.gate'
            peft_config.target_parameters.add(f"{target}.weight")
    # remove gate from target_modules
    peft_config.target_modules = {
        key for key in peft_config.target_modules if not ((key == "gate") or (key.endswith(".gate")))
    }

    # add expert layers: w1 & w3 => gate_up_proj, ModuleList of layers is now a stacked parameter.
    for target in peft_config.target_modules:
        # if only w1 or only w3 are targeted, conversion is not possible
        if (target == "w1") or target.endswith(".w1"):
            if target.replace("w1", "w3") not in peft_config.target_modules:
                raise ValueError("Cannot convert because blabla")  # FIXME
        if (target == "w3") or target.endswith(".w3"):
            if target.replace("w3", "w1") not in peft_config.target_modules:
                raise ValueError("Cannot convert because blabla")  # FIXME

        if (target == "w1") or target.endswith(".w1"):
            # FIXME: what if only specific layers are targeted, e.g. '0.w1'
            peft_config.target_parameters.add("gate_up_proj")
    # remove w1 and w3
    peft_config.target_modules = {
        key for key in peft_config.target_modules if not ((key == "w1") or (key.endswith(".w1")))
    }
    peft_config.target_modules = {
        key for key in peft_config.target_modules if not ((key == "w3") or (key.endswith(".w3")))
    }

    # add expert layers: w2 => down_proj, ModuleList of layers is now a stacked parameter.
    for target in peft_config.target_modules:
        if (target == "w2") or target.endswith(".w2"):
            # FIXME: what if only specific layers are targeted, e.g. '0.w2'
            peft_config.target_parameters.add("down_proj")
    # remove w1 and w3
    peft_config.target_modules = {
        key for key in peft_config.target_modules if not ((key == "w2") or (key.endswith(".w2")))
    }

    if 'gate_up_proj' in peft_config.target_parameters:
        # this weight is a fusion of two adapters so the internal representation is r*2
        peft_config.rank_pattern[r'.*\.experts\.gate_up_proj'] = peft_config.r * 2

    return peft_config


def convert_peft_config_for_transformers(peft_config, model: torch.nn.Module, conversions: list[Any] | None):
    # FIXME document this properly
    # Deal with weight conversion from transformers

    ##############################
    # check if conversion needed #
    ##############################

    # If, for any reason, we cannot apply conversion, we just return the PEFT config as is.
    from peft import PeftType  # avoid circular import

    if peft_config.peft_type != PeftType.LORA:
        # weight conversion is currently only supported for LoRA
        return peft_config
    if not hasattr(model, "config"):
        # not a transformer model
        return peft_config
    if not hasattr(model.config, "model_type"):
        # not a transformer model
        return peft_config

    # TODO: deal with general renamings

    ##########################
    # model specific changes #
    ##########################

    peft_config = copy.deepcopy(peft_config)  # don't mutate the original config

    # TODO So far, only dealing with Mixtral
    if model.config.model_type == "mixtral":
        peft_config = _convert_peft_config_mixtral(peft_config)

    return peft_config
