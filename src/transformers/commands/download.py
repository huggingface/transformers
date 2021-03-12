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

from argparse import ArgumentParser

from . import BaseTransformersCLICommand


def download_command_factory(args):
    return DownloadCommand(args.model, args.cache_dir, args.force)


class DownloadCommand(BaseTransformersCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        download_parser = parser.add_parser("download")
        download_parser.add_argument(
            "--cache-dir", type=str, default=None, help="Path to location to store the models"
        )
        download_parser.add_argument(
            "--force", action="store_true", help="Force the model to be download even if already in cache-dir"
        )
        download_parser.add_argument("model", type=str, help="Name of the model to download")
        download_parser.set_defaults(func=download_command_factory)

    def __init__(self, model: str, cache: str, force: bool):
        self._model = model
        self._cache = cache
        self._force = force

    def run(self):
        from ..models.auto import AutoModel, AutoTokenizer

        AutoModel.from_pretrained(self._model, cache_dir=self._cache, force_download=self._force)
        AutoTokenizer.from_pretrained(self._model, cache_dir=self._cache, force_download=self._force)
