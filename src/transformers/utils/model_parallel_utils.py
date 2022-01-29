# coding=utf-8
# Copyright 2020 The HuggingFace Team. All rights reserved.
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
import importlib
from math import ceil


class TPInfo(object):
    def __init__(
        self,
        *name,
        fuse: bool = False,
        reverse: bool = False,
    ):
        self.name = name
        self.fuse = fuse
        self.reverse = reverse

    def __str__(self):
        return f"{self.__class__.__qualname__}({self.name})"

    def __repr__(self):
        return self.__str__()


Col = type("COLUMN", (TPInfo,), {"code": "Col"})
Row = type("ROW", (TPInfo,), {"code": "Row"})
Update = type("UPDATE", (TPInfo,), {"code": "Update", "parallel": False})


class TPMapping(object):
    __MAPPING__ = dict(
        Albert=[
            Col("query", "key", "value", "ffn"),
            Row("dense", "ffn_output"),
            Update("num_attention_heads", "all_head_size"),
        ],
        Bart=[
            Col("q_proj", "k_proj", "v_proj", "fc1"),
            Row("out_proj", "fc2"),
            Update("embed_dim", "num_heads"),
        ],
        Bert=[
            Col("query", "key", "value", "intermediate.dense"),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
        ],
        T5=[
            Col("Attention.q", "Attention.k", "Attention.v"),
            Col("relative_attention_bias", reverse=True),
            Row("DenseReluDense.wi", "Attention.o", "DenseReluDense.wo"),
            Update("d_model", "n_heads", "inner_dim"),
        ],
        GPT2=[
            Col("c_attn", reverse=True, fuse=True),
            Col("q_attn", reverse=True),
            Row("c_proj", reverse=True),
            Update("embed_dim", "split_size", "num_heads"),
        ],
        Electra=[
            Col("query", "key", "value", "intermediate.dense"),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
        ],
        Roberta=[
            Col("query", "key", "value", "intermediate.dense"),
            Row("output.dense"),
            Update("num_attention_heads", "all_head_size"),
        ],
    )

    def __init__(self):
        cache_tp_mapping = {}

        for cls_name, mapping in self.__MAPPING__.items():
            cls = self._load_class_by_model_name(cls_name)
            cache_tp_mapping[cls] = []

            for elem in mapping:
                for name in elem.name:
                    copy_elem = copy.deepcopy(elem)
                    copy_elem.name = name
                    cache_tp_mapping[cls].append(copy_elem)

        self.__MAPPING__ = {cls: {} for cls in cache_tp_mapping}
        # clear exist mapping rather than making new mapping dict

        for cls, mapping in cache_tp_mapping.items():
            for elem in mapping:
                if elem.code in self.__MAPPING__[cls]:
                    self.__MAPPING__[cls][elem.code].append(elem)
                else:
                    self.__MAPPING__[cls][elem.code] = [elem]

    @staticmethod
    def _load_class_by_model_name(model_name):
        transformers = importlib.import_module("transformers")
        cls = getattr(transformers, f"{model_name}PreTrainedModel", None)
        if cls is None:
            cls = getattr(transformers, f"{model_name}PretrainedModel", None)
        assert cls is not None, f"Can not import the model named {cls}."
        return cls

    def get_mapping(self, model):
        for cls, mapping in self.__MAPPING__.items():
            if isinstance(model, cls):
                return mapping
        return None

    def column_parallel_params(self, model):
        mapping = self.get_mapping(model)
        if mapping is not None:
            return mapping["Col"]

    def row_parallel_params(self, model):
        mapping = self.get_mapping(model)
        if mapping is not None:
            return mapping["Row"]

    def update_attrs(self, model):
        mapping = self.get_mapping(model)
        if mapping is not None:
            return mapping["Update"]

    def search(self, model, param_name):
        mapping = self.get_mapping(model)
        count_contain_elem_in_param = 0
        param_split = param_name.split(".")
        first_check = []

        for code, elems in mapping.items():
            for elem in elems:
                if elem.name in param_name:
                    first_check.append(elem)

        for elem in first_check:
            elem_split = elem.name.split(".")
            for split in elem_split:
                if split in param_split:
                    count_contain_elem_in_param += 1
            if count_contain_elem_in_param == len(elem_split):
                return elem

        return None

    def is_fused_param(self, model, param_name):
        elem = self.search(model, param_name)
        if elem is not None:
            return elem.fuse

    def get_fusion_degree(self, model, param_name, module):
        if self.is_fused_param(model, param_name) and hasattr(module, "weight"):
            bigger = max(module.weight.size(0), module.weight.size(1))
            smaller = min(module.weight.size(0), module.weight.size(1))
            return bigger // smaller
        return 1

    def is_reversed_param(self, model, param_name):
        elem = self.search(model, param_name)
        if elem is not None:
            return elem.reverse

    def is_column_parallel(self, model, param_name):
        elem = self.search(model, param_name)
        if elem is not None:
            return elem.code == "Col"

    def is_row_parallel(self, model, param_name):
        elem = self.search(model, param_name)
        if elem is not None:
            return elem.code == "Row"


def assert_device_map(device_map, num_blocks):
    blocks = list(range(0, num_blocks))

    device_map_blocks = [item for sublist in list(device_map.values()) for item in sublist]

    # Duplicate check
    duplicate_blocks = []
    for i in device_map_blocks:
        if device_map_blocks.count(i) > 1 and i not in duplicate_blocks:
            duplicate_blocks.append(i)
    # Missing blocks
    missing_blocks = [i for i in blocks if i not in device_map_blocks]
    extra_blocks = [i for i in device_map_blocks if i not in blocks]

    if len(duplicate_blocks) != 0:
        raise ValueError(
            "Duplicate attention blocks specified in device_map. Attention blocks must be specified to one device. These "
            "attention blocks were specified more than once: " + str(duplicate_blocks)
        )
    if len(missing_blocks) != 0:
        raise ValueError(
            "There are attention blocks for this model that are not specified in the device_map. Add these attention "
            "blocks to a device on the device_map: " + str(missing_blocks)
        )
    if len(extra_blocks) != 0:
        raise ValueError(
            "The device_map contains more attention blocks than this model has. Remove these from the device_map:"
            + str(extra_blocks)
        )


def get_device_map(n_layers, devices):
    """Returns a dictionary of layers distributed evenly across all devices."""
    layers = list(range(n_layers))
    n_blocks = int(ceil(n_layers / len(devices)))
    layers_list = list(layers[i : i + n_blocks] for i in range(0, n_layers, n_blocks))

    return dict(zip(devices, layers_list))
