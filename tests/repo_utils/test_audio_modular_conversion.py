# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT / "utils"))

from check_modular_conversion import convert_and_run_ruff  # noqa: E402


class AudioModularConversionTest(unittest.TestCase):
    def test_regeneration_preserves_all_generated_files(self):
        for model, parent in (("neucodec", "xcodec2"), ("nemotron_asr_streaming", "parakeet")):
            model_dir = ROOT / "src/transformers/models" / model
            files = convert_and_run_ruff(str(model_dir / f"modular_{model}.py"))
            self.assertEqual(set(files), {"configuration", "modeling", "audio_processing", "audio_processing_numpy"})
            for kind, code in files.items():
                with self.subTest(model=model, kind=kind):
                    self.assertEqual(code, (model_dir / f"{kind}_{model}.py").read_text())
                    if kind.startswith("audio_processing"):
                        self.assertNotIn(f"from ..{parent}", code)

    def test_numpy_processor_runs_without_torch(self):
        code = """
import sys
sys.modules['torch'] = None
import numpy as np
from transformers.models.neucodec.audio_processing_numpy_neucodec import NeuCodecAudioProcessorNumpy
output = NeuCodecAudioProcessorNumpy()(
    np.random.RandomState(0).randn(1600).astype(np.float32), return_tensors='np'
)
assert output['audio_values'].shape == (1, 1, 1920)
assert output['audio_features'].shape == (1, 5, 160)
assert output['audio_features_mask'].shape == (1, 5)
from transformers.models.nemotron_asr_streaming.audio_processing_numpy_nemotron_asr_streaming import NemotronAsrStreamingAudioProcessorNumpy
streaming = NemotronAsrStreamingAudioProcessorNumpy()(np.zeros(1600, dtype=np.float32), return_tensors='np')
assert streaming['audio_features'].shape[:2] == streaming['audio_features_mask'].shape
assert sys.modules['torch'] is None
"""
        env = {**os.environ, "USE_TORCH": "0", "PYTHONPATH": str(ROOT / "src")}
        result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
