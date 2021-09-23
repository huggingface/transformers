# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team.
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

import argparse
import glob
import json
import os
import tarfile

from datasets.utils.file_utils import http_get

from transformers import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

DATASET_URL = "https://zenodo.org/record/5036977/files/CommonLanguage.tar.gz?download=1"


def prepare_commonlanguage(download_directory):
    """
    Download and convert the CommonLanguage dataset for audio classification
    """
    download_directory = os.path.abspath(download_directory)
    os.makedirs(download_directory, exist_ok=True)
    archive_path = os.path.join(download_directory, "CommonLanguage.tar.gz")

    if not os.path.exists(archive_path):
        logger.info(f"Downloading the dataset to {archive_path}")
        with open(archive_path, "wb") as fout:
            http_get(DATASET_URL, fout)

    logger.info(f"Unpacking the dataset...")
    with tarfile.open(archive_path) as tar_file:
        tar_file.extractall(download_directory)

    for split in ("train", "dev", "test"):
        logger.info(f"Gathering '{split}' wav paths...")
        wav_glob = os.path.join(download_directory, f"common_voice_kpd/*/{split}/*/*.wav")
        wav_paths = sorted(glob.glob(wav_glob))
        dataset_file = os.path.join(download_directory, f"{split}.json")
        with open(dataset_file, "w") as fout:
            for path in wav_paths:
                example = {"file": path, "label": path.split("/")[-4]}
                fout.write(json.dumps(example) + "\n")
        logger.info(f"Wrote the dataset split to '{dataset_file}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_directory", default="./CommonLanguage", type=str, help="Path to the dataset root")
    args = parser.parse_args()
    prepare_commonlanguage(args.download_directory)
