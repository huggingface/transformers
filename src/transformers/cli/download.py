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
from typing import Annotated, Optional

import typer


def download(
    model_id: Annotated[str, typer.Argument(help="The model ID to download")],
    cache_dir: Annotated[Optional[str], typer.Option(help="Directory where to save files.")] = None,
    force_download: Annotated[
        bool, typer.Option(help="If set, the files will be downloaded even if they are already cached locally.")
    ] = False,
    trust_remote_code: Annotated[
        bool,
        typer.Option(
            help="Whether or not to allow for custom models defined on the Hub in their own modeling files. Use only if you've reviewed the code as it will execute on your local machine"
        ),
    ] = False,
):
    """Download a model and its tokenizer from the Hub."""
    from ..models.auto import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(
        model_id, cache_dir=cache_dir, force_download=force_download, trust_remote_code=trust_remote_code
    )
    AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, force_download=force_download, trust_remote_code=trust_remote_code
    )
