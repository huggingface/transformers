# Copyright 2025 HuggingFace Inc.
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

import json
import os
import tempfile

import numpy as np

from transformers.testing_utils import (
    check_json_file_has_correct_format,
    require_torch,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch


def prepare_audio_inputs(
    batch_size,
    min_length=400,
    max_length=2000,
    num_channels=1,
    equal_length=False,
    numpify=False,
    torchify=False,
):
    """This function prepares a list of numpy arrays, or a list of PyTorch tensors if one specifies torchify=True.

    One can specify whether the audio inputs are of the same length or not.
    """

    assert not (numpify and torchify), "You cannot specify both numpy and PyTorch tensors at the same time"

    audio_inputs = []
    for _ in range(batch_size):
        if equal_length:
            length = max_length
        else:
            length = np.random.randint(min_length, max_length)

        if num_channels > 1:
            audio_inputs.append(np.random.randn(length, num_channels).astype(np.float32))
        else:
            audio_inputs.append(np.random.randn(length).astype(np.float32))

    if torchify:
        audio_inputs = [torch.from_numpy(audio) for audio in audio_inputs]

    return audio_inputs


class AudioProcessingTestMixin:
    """Mixin class for testing audio processors, analogous to ``ImageProcessingTestMixin``.

    Subclasses must set the following in ``setUp``:

    * ``self.audio_processing_classes``  – ``dict[str, type]`` mapping backend name to class
    * ``self.audio_processor_dict``      – kwargs to instantiate the processor
    * ``self.audio_processor_tester``    – object with ``prepare_audio_inputs()`` and ``batch_size``
    """

    # ─── Serialization ────────────────────────────────────────────────

    def test_audio_processor_to_json_string(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processor = audio_processing_class(**self.audio_processor_dict)
            obj = json.loads(audio_processor.to_json_string())
            for key, value in self.audio_processor_dict.items():
                self.assertEqual(obj[key], value)

    def test_audio_processor_to_json_file(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processor_first = audio_processing_class(**self.audio_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                json_file_path = os.path.join(tmpdirname, "audio_processor.json")
                audio_processor_first.to_json_file(json_file_path)
                audio_processor_second = audio_processing_class.from_json_file(json_file_path)

            self.assertEqual(audio_processor_second.to_dict(), audio_processor_first.to_dict())

    def test_audio_processor_from_and_save_pretrained(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processor_first = audio_processing_class(**self.audio_processor_dict)

            with tempfile.TemporaryDirectory() as tmpdirname:
                saved_file = audio_processor_first.save_pretrained(tmpdirname)[0]
                check_json_file_has_correct_format(saved_file)
                audio_processor_second = audio_processing_class.from_pretrained(tmpdirname)

            self.assertEqual(audio_processor_second.to_dict(), audio_processor_first.to_dict())

    def test_init_without_params(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processor = audio_processing_class()
            self.assertIsNotNone(audio_processor)

    # ─── Backend equivalence ──────────────────────────────────────────

    @require_torch
    def test_backends_equivalence(self):
        if len(self.audio_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        audio_input = np.random.randn(16000).astype(np.float32)
        sample_rate = self.audio_processor_dict.get("sample_rate", 16000)

        encodings = {}
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processor = audio_processing_class(**self.audio_processor_dict)
            encodings[backend_name] = audio_processor(audio_input, sample_rate=sample_rate, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_key = list(encodings[reference_backend].keys())[0]
        reference_values = encodings[reference_backend][reference_key]
        for backend_name in backend_names[1:]:
            torch.testing.assert_close(reference_values, encodings[backend_name][reference_key], atol=1e-5, rtol=1e-5)

    @require_torch
    def test_backends_equivalence_batched(self):
        if len(self.audio_processing_classes) < 2:
            self.skipTest(reason="Skipping backends equivalence test as there are less than 2 backends")

        audio_inputs = self.audio_processor_tester.prepare_audio_inputs(equal_length=False)
        sample_rate = self.audio_processor_dict.get("sample_rate", 16000)

        encodings = {}
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processor = audio_processing_class(**self.audio_processor_dict)
            encodings[backend_name] = audio_processor(audio_inputs, sample_rate=sample_rate, return_tensors="pt")

        backend_names = list(encodings.keys())
        reference_backend = backend_names[0]
        reference_key = list(encodings[reference_backend].keys())[0]
        reference_values = encodings[reference_backend][reference_key]
        for backend_name in backend_names[1:]:
            torch.testing.assert_close(reference_values, encodings[backend_name][reference_key], atol=1e-5, rtol=1e-5)

    # ─── Cross-backend save / load ────────────────────────────────────

    def test_save_load_backends(self):
        """Test that we can load audio processors saved by one backend with another."""
        if len(self.audio_processing_classes) < 2:
            self.skipTest("Skipping backend save/load test as there are less than 2 backends")

        backend_names = list(self.audio_processing_classes.keys())

        for backend1 in backend_names:
            processor1 = self.audio_processing_classes[backend1](**self.audio_processor_dict)

            for backend2 in backend_names:
                if backend1 == backend2:
                    continue

                with tempfile.TemporaryDirectory() as tmpdirname:
                    processor1.save_pretrained(tmpdirname)
                    processor2 = self.audio_processing_classes[backend2].from_pretrained(tmpdirname)

                dict1 = processor1.to_dict()
                dict2 = processor2.to_dict()
                common_keys = set(dict1) & set(dict2)
                self.assertEqual(
                    {k: dict1[k] for k in common_keys},
                    {k: dict2[k] for k in common_keys},
                    f"Backends {backend1} and {backend2} differ in common keys",
                )

    # ─── Input type tests ─────────────────────────────────────────────

    @require_torch
    def test_call_numpy(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processing = audio_processing_class(**self.audio_processor_dict)
            audio_inputs = self.audio_processor_tester.prepare_audio_inputs(equal_length=False)
            for audio in audio_inputs:
                self.assertIsInstance(audio, np.ndarray)

            sample_rate = self.audio_processor_dict.get("sample_rate", 16000)

            # Test not batched input
            encoded = audio_processing(audio_inputs[0], sample_rate=sample_rate, return_tensors="pt")
            output_key = list(encoded.keys())[0]
            self.assertEqual(len(encoded[output_key].shape), 2)  # (1, length)

            # Test batched
            encoded = audio_processing(audio_inputs, sample_rate=sample_rate, return_tensors="pt")
            self.assertEqual(encoded[output_key].shape[0], self.audio_processor_tester.batch_size)

    @require_torch
    def test_call_pytorch(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processing = audio_processing_class(**self.audio_processor_dict)
            audio_inputs = self.audio_processor_tester.prepare_audio_inputs(equal_length=False, torchify=True)

            for audio in audio_inputs:
                self.assertIsInstance(audio, torch.Tensor)

            sample_rate = self.audio_processor_dict.get("sample_rate", 16000)

            # Test not batched input
            encoded = audio_processing(audio_inputs[0], sample_rate=sample_rate, return_tensors="pt")
            output_key = list(encoded.keys())[0]
            self.assertEqual(len(encoded[output_key].shape), 2)

            # Test batched
            encoded = audio_processing(audio_inputs, sample_rate=sample_rate, return_tensors="pt")
            self.assertEqual(encoded[output_key].shape[0], self.audio_processor_tester.batch_size)

    @require_torch
    def test_call_multichannel_force_mono(self):
        """Test that multi-channel audio is correctly averaged to mono."""
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            processor_dict = {**self.audio_processor_dict, "force_mono": True}
            audio_processing = audio_processing_class(**processor_dict)

            audio_inputs = prepare_audio_inputs(
                batch_size=self.audio_processor_tester.batch_size,
                num_channels=2,
                min_length=self.audio_processor_tester.min_length,
                max_length=self.audio_processor_tester.max_length,
                equal_length=True,
            )

            sample_rate = self.audio_processor_dict.get("sample_rate", 16000)
            encoded = audio_processing(audio_inputs, sample_rate=sample_rate, return_tensors="pt")
            output_key = list(encoded.keys())[0]
            # After force_mono, output should be 2D: (batch, length)
            self.assertEqual(len(encoded[output_key].shape), 2)

    # ─── Padding tests ────────────────────────────────────────────────

    @require_torch
    def test_padding_right(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            processor_dict = {**self.audio_processor_dict, "padding_side": "right"}
            audio_processing = audio_processing_class(**processor_dict)

            audio_inputs = [
                np.random.randn(100).astype(np.float32),
                np.random.randn(200).astype(np.float32),
            ]
            sample_rate = self.audio_processor_dict.get("sample_rate", 16000)
            encoded = audio_processing(audio_inputs, sample_rate=sample_rate, return_tensors="pt")
            output_key = list(encoded.keys())[0]
            self.assertEqual(encoded[output_key].shape[-1], 200)

    @require_torch
    def test_padding_left(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            processor_dict = {**self.audio_processor_dict, "padding_side": "left"}
            audio_processing = audio_processing_class(**processor_dict)

            audio_inputs = [
                np.random.randn(100).astype(np.float32),
                np.random.randn(200).astype(np.float32),
            ]
            sample_rate = self.audio_processor_dict.get("sample_rate", 16000)
            encoded = audio_processing(audio_inputs, sample_rate=sample_rate, return_tensors="pt")
            output_key = list(encoded.keys())[0]
            self.assertEqual(encoded[output_key].shape[-1], 200)

    # ─── Truncation tests ─────────────────────────────────────────────

    @require_torch
    def test_truncation(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processing = audio_processing_class(**self.audio_processor_dict)

            audio_inputs = [
                np.random.randn(500).astype(np.float32),
                np.random.randn(1000).astype(np.float32),
            ]
            sample_rate = self.audio_processor_dict.get("sample_rate", 16000)
            encoded = audio_processing(
                audio_inputs, sample_rate=sample_rate, truncation=True, max_length=300, return_tensors="pt"
            )
            output_key = list(encoded.keys())[0]
            self.assertEqual(encoded[output_key].shape[-1], 300)

    @require_torch
    def test_truncation_without_max_length_raises(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processing = audio_processing_class(**self.audio_processor_dict)

            audio_inputs = [np.random.randn(500).astype(np.float32)]
            sample_rate = self.audio_processor_dict.get("sample_rate", 16000)
            with self.assertRaises(ValueError):
                audio_processing(
                    audio_inputs, sample_rate=sample_rate, truncation=True, max_length=None, return_tensors="pt"
                )

    # ─── pad_to_multiple_of ───────────────────────────────────────────

    @require_torch
    def test_pad_to_multiple_of(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processing = audio_processing_class(**self.audio_processor_dict)

            audio_inputs = [np.random.randn(100).astype(np.float32)]
            sample_rate = self.audio_processor_dict.get("sample_rate", 16000)
            encoded = audio_processing(
                audio_inputs,
                sample_rate=sample_rate,
                truncation=True,
                max_length=150,
                pad_to_multiple_of=64,
                return_tensors="pt",
            )
            output_key = list(encoded.keys())[0]
            # max_length=150 rounded up to next multiple of 64 → 192
            self.assertEqual(encoded[output_key].shape[-1] % 64, 0)

    # ─── Sample rate validation ───────────────────────────────────────

    def test_wrong_sample_rate_raises(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processing = audio_processing_class(**self.audio_processor_dict)

            audio_inputs = [np.random.randn(100).astype(np.float32)]
            expected_sr = self.audio_processor_dict.get("sample_rate", 16000)
            with self.assertRaises(ValueError):
                audio_processing(audio_inputs, sample_rate=expected_sr + 1000, return_tensors="pt")

    # ─── Dtype casting ────────────────────────────────────────────────

    @require_torch
    def test_cast_dtype(self):
        for backend_name, audio_processing_class in self.audio_processing_classes.items():
            audio_processing = audio_processing_class(**self.audio_processor_dict)

            audio_inputs = self.audio_processor_tester.prepare_audio_inputs(equal_length=True)
            sample_rate = self.audio_processor_dict.get("sample_rate", 16000)

            encoding = audio_processing(audio_inputs, sample_rate=sample_rate, return_tensors="pt")
            output_key = list(encoding.keys())[0]
            self.assertEqual(encoding[output_key].dtype, torch.float32)

            encoding = encoding.to(torch.float16)
            self.assertEqual(encoding[output_key].dtype, torch.float16)

            encoding = audio_processing(audio_inputs, sample_rate=sample_rate, return_tensors="pt").to(
                "cpu", torch.bfloat16
            )
            self.assertEqual(encoding[output_key].device, torch.device("cpu"))
            self.assertEqual(encoding[output_key].dtype, torch.bfloat16)
