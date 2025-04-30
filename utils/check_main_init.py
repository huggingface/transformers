# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
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
"""
Utility that checks whether the main init correctly contains all the models so that static type checkers work correctly.
"""

import argparse
from pathlib import Path

from transformers.utils.import_utils import create_import_structure_from_path, spread_import_structure


TRANSFORMERS_PATH = Path("src/transformers")
INIT_FILE = TRANSFORMERS_PATH / "__init__.py"


def check_models_in_init(fix_and_overwrite):
    init_content = INIT_FILE.read_text()
    prefix, model_specific_content_and_suffix = init_content.split("from .models import (", maxsplit=1)
    model_specific_content, suffix = model_specific_content_and_suffix.split(")", maxsplit=1)
    model_objects = [model.strip() for model in model_specific_content.split(",\n") if len(model.strip())]

    import_structure = create_import_structure_from_path(TRANSFORMERS_PATH)
    models = import_structure["models"]
    models = spread_import_structure(models)

    all_objects = set()

    for _, dependency in models.items():
        for k, v in dependency.items():
            all_objects.update([vv for vv in v if len(vv)])

    if fix_and_overwrite:
        all_uppercase = sorted({o for o in all_objects if o.isupper()}, key=str.casefold)
        all_lowercase = sorted({o for o in all_objects if o.islower()}, key=str.casefold)
        all_others = sorted(all_objects - set(all_uppercase + all_lowercase), key=str.casefold)

        all_uppercase = ",\n        ".join(all_uppercase)
        all_lowercase = ",\n        ".join(all_lowercase)
        all_others = ",\n        ".join(all_others)

        result = f"""from .models import (
        {all_uppercase},
        {all_others},
        {all_lowercase},
    )"""
        init_content = "".join([prefix, result, suffix])
        INIT_FILE.write_text(init_content)
    else:
        if set(all_objects) != set(model_objects):
            missing_objects = set(all_objects) - set(model_objects)
            objects_defined_but_non_existent = set(model_objects) - set(all_objects)

            error = "The imported model objects in __init__.py do not correspond to model objects defined in the repo."

            if len(missing_objects):
                error += f"\nThe following objects are missing from the main init: {missing_objects}"

            if len(objects_defined_but_non_existent):
                error += (
                    f"\nThe following objects are defined in the main init: {objects_defined_but_non_existent} but "
                    f"do not exist in the repo."
                )

            error += "\n\nRun make fix-copies to fix this."

            raise ValueError(error)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fix_and_overwrite", action="store_true", help="Whether to fix inconsistencies.")
    args = parser.parse_args()

    check_models_in_init(args.fix_and_overwrite)
