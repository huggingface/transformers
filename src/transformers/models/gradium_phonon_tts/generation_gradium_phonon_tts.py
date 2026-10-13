# Copyright 2026 Gradium and The HuggingFace Inc. team. All rights reserved.
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

from ..pocket_tts.generation_pocket_tts import PocketTTSGenerationMixin


# Gradium Phonon TTS generates the way Pocket TTS does; its runtime values (`eos_threshold`, `frames_after_eos`) come from
# the checkpoint's generation config and processor config.
class GradiumPhononTTSGenerationMixin(PocketTTSGenerationMixin):
    def _placeholder_token_ids(self) -> list[int]:
        # no voice BOS
        return [self.config.audio_prompt_token_id, self.config.audio_bos_token_id, self.config.audio_latent_token_id]
