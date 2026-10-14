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

import os
import shutil
import tempfile
import unittest

import numpy as np

from transformers import AutoTokenizer, BarkProcessor
from transformers.testing_utils import require_torch, slow


@require_torch
class BarkProcessorTest(unittest.TestCase):
    def setUp(self):
        self.checkpoint = "suno/bark-small"
        self.tmpdirname = tempfile.mkdtemp()
        self.voice_preset = "en_speaker_1"
        self.input_string = "This is a test string"
        self.speaker_embeddings_dict_path = "speaker_embeddings_path.json"
        self.speaker_embeddings_directory = "speaker_embeddings"

    def get_tokenizer(self, **kwargs):
        return AutoTokenizer.from_pretrained(self.checkpoint, **kwargs)

    def tearDown(self):
        shutil.rmtree(self.tmpdirname)

    def test_save_load_pretrained_default(self):
        tokenizer = self.get_tokenizer()

        processor = BarkProcessor(tokenizer=tokenizer)

        processor.save_pretrained(self.tmpdirname)
        processor = BarkProcessor.from_pretrained(self.tmpdirname)

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer.get_vocab())

    @slow
    def test_save_load_pretrained_additional_features(self):
        processor = BarkProcessor.from_pretrained(
            pretrained_processor_name_or_path=self.checkpoint,
            speaker_embeddings_dict_path=self.speaker_embeddings_dict_path,
        )

        # TODO (ebezzam) not all speaker embedding are properly downloaded.
        # My hypothesis: there are many files (~700 speaker embeddings) and some fail to download (not the same at different first runs)
        # https://github.com/huggingface/transformers/blob/967045082faaaaf3d653bfe665080fd746b2bb60/src/transformers/models/bark/processing_bark.py#L89
        # https://github.com/huggingface/transformers/blob/967045082faaaaf3d653bfe665080fd746b2bb60/src/transformers/models/bark/processing_bark.py#L188
        # So for testing purposes, we will remove the unavailable speaker embeddings before saving.
        processor._verify_speaker_embeddings(remove_unavailable=True)
        processor.save_pretrained(
            self.tmpdirname,
            speaker_embeddings_dict_path=self.speaker_embeddings_dict_path,
            speaker_embeddings_directory=self.speaker_embeddings_directory,
        )

        tokenizer_add_kwargs = self.get_tokenizer(bos_token="(BOS)", eos_token="(EOS)")

        processor = BarkProcessor.from_pretrained(
            self.tmpdirname,
            self.speaker_embeddings_dict_path,
            bos_token="(BOS)",
            eos_token="(EOS)",
        )

        self.assertEqual(processor.tokenizer.get_vocab(), tokenizer_add_kwargs.get_vocab())

    def test_speaker_embeddings(self):
        processor = BarkProcessor.from_pretrained(
            pretrained_processor_name_or_path=self.checkpoint,
            speaker_embeddings_dict_path=self.speaker_embeddings_dict_path,
        )

        seq_len = 35
        nb_codebooks_coarse = 2
        nb_codebooks_total = 8

        voice_preset = {
            "semantic_prompt": np.ones(seq_len),
            "coarse_prompt": np.ones((nb_codebooks_coarse, seq_len)),
            "fine_prompt": np.ones((nb_codebooks_total, seq_len)),
        }

        # test providing already loaded voice_preset
        inputs = processor(text=self.input_string, voice_preset=voice_preset)

        processed_voice_preset = inputs["history_prompt"]
        for key in voice_preset:
            self.assertListEqual(voice_preset[key].tolist(), processed_voice_preset.get(key, np.array([])).tolist())

        # test loading voice preset from npz file
        tmpfilename = os.path.join(self.tmpdirname, "file.npz")
        np.savez(tmpfilename, **voice_preset)
        inputs = processor(text=self.input_string, voice_preset=tmpfilename)
        processed_voice_preset = inputs["history_prompt"]

        for key in voice_preset:
            self.assertListEqual(voice_preset[key].tolist(), processed_voice_preset.get(key, np.array([])).tolist())

        # test loading voice preset from the hub
        inputs = processor(text=self.input_string, voice_preset=self.voice_preset)

    def test_speaker_embeddings_saving_rejects_path_traversal(self):
        # A malicious speaker_embeddings_path.json dict key must not be usable to escape the save
        # directory and write attacker-controlled content to an arbitrary path (path traversal, CWE-22).
        tokenizer = self.get_tokenizer()
        seq_len = 5
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            # Plant the per-prompt npy files the malicious "repo" claims to provide so that
            # `_load_voice_preset` succeeds and we reach the vulnerable `np.save` call.
            np.save(os.path.join(tmp_dir_name, "s.npy"), np.ones(seq_len))
            np.save(os.path.join(tmp_dir_name, "c.npy"), np.ones((2, seq_len)))
            np.save(os.path.join(tmp_dir_name, "f.npy"), np.ones((8, seq_len)))
            speaker_embeddings = {
                "repo_or_path": tmp_dir_name,
                "../../PWNED": {"semantic_prompt": "s.npy", "coarse_prompt": "c.npy", "fine_prompt": "f.npy"},
            }
            processor = BarkProcessor(tokenizer=tokenizer, speaker_embeddings=speaker_embeddings)

            save_dir = os.path.join(tmp_dir_name, "save")
            # Where the "../../PWNED" key would land if traversal succeeded.
            canary = os.path.join(tmp_dir_name, "PWNED_semantic_prompt.npy")
            with self.assertRaises(ValueError):
                processor.save_pretrained(save_dir)
            self.assertFalse(os.path.exists(canary))

    def test_speaker_embeddings_saving_allows_subdirectories(self):
        # Voice presets are legitimately stored in subdirectories (e.g. the `v2/...` presets in
        # ylacombe/bark-large), so saving a nested key must be allowed - the guard rejects escapes only.
        tokenizer = self.get_tokenizer()
        seq_len = 5
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            np.save(os.path.join(tmp_dir_name, "s.npy"), np.ones(seq_len))
            np.save(os.path.join(tmp_dir_name, "c.npy"), np.ones((2, seq_len)))
            np.save(os.path.join(tmp_dir_name, "f.npy"), np.ones((8, seq_len)))
            speaker_embeddings = {
                "repo_or_path": tmp_dir_name,
                "v2/en_speaker_0": {"semantic_prompt": "s.npy", "coarse_prompt": "c.npy", "fine_prompt": "f.npy"},
            }
            processor = BarkProcessor(tokenizer=tokenizer, speaker_embeddings=speaker_embeddings)

            save_dir = os.path.join(tmp_dir_name, "save")
            processor.save_pretrained(save_dir)
            self.assertTrue(
                os.path.exists(os.path.join(save_dir, "speaker_embeddings", "v2", "en_speaker_0_semantic_prompt.npy"))
            )

    def test_load_voice_preset_rejects_path_traversal(self):
        # The per-prompt paths in speaker_embeddings_path.json are also untrusted and are joined onto
        # repo_or_path before being read, so a "../x" value must be rejected before the file is loaded.
        tokenizer = self.get_tokenizer()
        seq_len = 5
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            repo_dir = os.path.join(tmp_dir_name, "repo")
            os.makedirs(repo_dir)
            # A readable .npy outside the repo dir that the traversal would otherwise reach.
            np.save(os.path.join(tmp_dir_name, "PWNED.npy"), np.ones(seq_len))
            speaker_embeddings = {
                "repo_or_path": repo_dir,
                "evil": {
                    "semantic_prompt": "../PWNED.npy",
                    "coarse_prompt": "../PWNED.npy",
                    "fine_prompt": "../PWNED.npy",
                },
            }
            processor = BarkProcessor(tokenizer=tokenizer, speaker_embeddings=speaker_embeddings)
            with self.assertRaises(ValueError):
                processor._load_voice_preset("evil")

    def test_load_voice_preset_allows_symlinked_cache_files(self):
        # The path-traversal guard must be lexical, not symlink-resolving: the HF hub cache stores each
        # snapshot file as a symlink into a sibling `blobs/` dir (snapshots/<rev>/f -> ../../blobs/<sha>),
        # which sits outside repo_or_path. A resolve()/realpath()-based check would follow that symlink
        # out of repo_or_path and wrongly reject a legitimate load, so loading must still succeed here.
        tokenizer = self.get_tokenizer()
        seq_len = 5
        with tempfile.TemporaryDirectory() as tmp_dir_name:
            # Mimic the hub cache layout: models--org--model/{blobs,snapshots/<rev>/...}.
            repo_cache = os.path.join(tmp_dir_name, "models--dummy--bark")
            blobs_dir = os.path.join(repo_cache, "blobs")
            snapshot_dir = os.path.join(repo_cache, "snapshots", "deadbeef")
            os.makedirs(blobs_dir)
            os.makedirs(snapshot_dir)

            arrays = {
                "semantic_prompt": np.ones(seq_len),
                "coarse_prompt": np.ones((2, seq_len)),
                "fine_prompt": np.ones((8, seq_len)),
            }
            voice_preset_paths = {}
            for key, array in arrays.items():
                blob = os.path.join(blobs_dir, key)  # real content lives in blobs/
                np.save(blob, array, allow_pickle=False)
                # ...and the snapshot exposes it as a relative symlink, exactly like the real cache.
                link = os.path.join(snapshot_dir, f"{key}.npy")
                os.symlink(os.path.relpath(blob + ".npy", snapshot_dir), link)
                voice_preset_paths[key] = f"{key}.npy"
            self.assertTrue(os.path.islink(os.path.join(snapshot_dir, "semantic_prompt.npy")))

            speaker_embeddings = {"repo_or_path": snapshot_dir, "preset": voice_preset_paths}
            processor = BarkProcessor(tokenizer=tokenizer, speaker_embeddings=speaker_embeddings)

            voice_preset = processor._load_voice_preset("preset")
            for key, array in arrays.items():
                self.assertTrue(np.array_equal(voice_preset[key], array))

    def test_tokenizer(self):
        tokenizer = self.get_tokenizer()

        processor = BarkProcessor(tokenizer=tokenizer)

        encoded_processor = processor(text=self.input_string)

        encoded_tok = tokenizer(
            self.input_string,
            padding="max_length",
            max_length=256,
            add_special_tokens=False,
            return_attention_mask=True,
            return_token_type_ids=False,
        )

        for key in encoded_tok:
            self.assertListEqual(encoded_tok[key], encoded_processor[key].squeeze().tolist())
