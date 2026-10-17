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


from transformers.testing_utils import require_torch, require_vision
from transformers.utils import is_vision_available

from ..idefics3 import test_image_processing_idefics3


if is_vision_available():
    from PIL import Image


class UltraVBertImageProcessingTester(test_image_processing_idefics3.Idefics3ImageProcessingTester):
    pass


@require_torch
@require_vision
class UltraVBertImageProcessingTest(test_image_processing_idefics3.Idefics3ImageProcessingTest):
    image_processor_tester_class = UltraVBertImageProcessingTester

    def test_max_image_tiles_lowers_the_longest_edge(self):
        page = Image.new("RGB", (1700, 2200), "white")
        sizes = {"size": {"longest_edge": 3072}, "max_image_size": {"longest_edge": 512}}
        for image_processing_class in self.image_processing_classes.values():
            uncapped = image_processing_class(**sizes)
            self.assertEqual(uncapped.get_number_of_image_patches(2200, 1700), (31, 6, 5))

            capped = image_processing_class(**sizes, max_image_tiles=27)
            outputs = capped([[page]], return_row_col_info=True, return_tensors="pt")
            self.assertEqual((outputs["rows"][0][0], outputs["cols"][0][0]), (5, 4))
            self.assertEqual(outputs["pixel_values"].shape[1], 21)
            self.assertEqual(capped.get_number_of_image_patches(2200, 1700), (21, 5, 4))
