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
from typing import Annotated

import typer


def download(
    model_id: Annotated[str, typer.Argument(help="Model ID to download")],
    cache_dir: Annotated[str | None, typer.Option(help="Directory to store files")] = None,
    force_download: Annotated[bool, typer.Option(help="Download files even if they're already cached")] = False,
    trust_remote_code: Annotated[
        bool,
        typer.Option(help="Allow custom model code from the Hub to run locally after you've reviewed it"),
    ] = False,
):
    """Download a model and tokenizer from the Hugging Face Hub"""
    from ..models.auto import AutoModel, AutoTokenizer

    AutoModel.from_pretrained(
        model_id, cache_dir=cache_dir, force_download=force_download, trust_remote_code=trust_remote_code
    )
    AutoTokenizer.from_pretrained(
        model_id, cache_dir=cache_dir, force_download=force_download, trust_remote_code=trust_remote_code
    )
