# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team.
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

from get_test_info import get_tester_classes


if __name__ == "__main__":
    failures = []

    model_test_root_dir = os.path.join("tests", "models")

    model_types = os.listdir(model_test_root_dir)
    model_types = [x for x in model_types if os.path.isdir(os.path.join(model_test_root_dir, x))]
    model_types = [x for x in model_types if x not in ["auto", "__pycache__"]]

    for model_type in model_types[:]:
        test_dir = os.path.join(model_test_root_dir, model_type)
        test_fns = os.listdir(test_dir)
        test_fns = [x for x in test_fns if x.startswith("test_modeling_")]
        # TODO: deal with TF/Flax too
        test_fns = [
            x for x in test_fns if not (x.startswith("test_modeling_tf_") or x.startswith("test_modeling_flax_"))
        ]
        for test_fn in test_fns:
            test_file = os.path.join(test_dir, test_fn)
            tester_classes = get_tester_classes(test_file)
            for tester_class in tester_classes:
                print(tester_class.__name__)
                # A few tester classes don't have `parent` parameter in `__init__`
                # TODO: deal this better
                try:
                    tester = tester_class(parent=None)
                except Exception:
                    continue
                if hasattr(tester, "get_config"):
                    config = tester.get_config()
                    for k, v in config.to_dict().items():
                        if isinstance(v, int) and v >= 100:

                            if k.endswith("_token_id"):
                                # skip
                                continue
                            elif k.endswith("max_position_embeddings"):
                                # TODO: 78 to fix
                                continue
                            elif k.endswith("hidden_size"):
                                # TODO: 21 to fix
                                continue
                            elif k.endswith("_dim"):
                                # TODO: 20 to fix
                                continue
                            elif k.endswith("_size"):
                                # TODO: 60 -21 = 39 to fix
                                continue

                            # Need to deal with (allow) special cases, otherwise change some values
                            failures.append(
                                f"{tester_class.__name__} will produce a `config` of type `{config.__class__.__name__}`"
                                f' with config["{k}"] = {v} which is too large for testing!'
                            )

    if len(failures) > 0:
        raise Exception(f"There were {len(failures)} failures:\n" + "\n".join(failures))
