# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import unittest

from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key


class Ovis2_5CheckpointConversionTest(unittest.TestCase):
    def test_vision_encoder_layers_map_to_flat_vision_layers(self):
        source_key = "visual_tokenizer.vit.vision_model.encoder.layers.0.self_attn.q_proj.weight"
        target_key = "model.vision_tower.layers.0.self_attn.q_proj.weight"
        conversions = get_checkpoint_conversion_mapping("ovis2_5")
        renamings = [conversion for conversion in conversions if isinstance(conversion, WeightRenaming)]
        converters = [conversion for conversion in conversions if isinstance(conversion, WeightConverter)]

        converted_key, _ = rename_source_key(source_key, renamings, converters)

        self.assertEqual(converted_key, target_key)

        reverse_conversions = [conversion.reverse_transform() for conversion in conversions[::-1]]
        reverse_renamings = [
            conversion for conversion in reverse_conversions if isinstance(conversion, WeightRenaming)
        ]
        reverse_converters = [
            conversion for conversion in reverse_conversions if isinstance(conversion, WeightConverter)
        ]
        restored_key, _ = rename_source_key(target_key, reverse_renamings, reverse_converters, reverse=True)

        self.assertEqual(restored_key, source_key)


if __name__ == "__main__":
    unittest.main()
