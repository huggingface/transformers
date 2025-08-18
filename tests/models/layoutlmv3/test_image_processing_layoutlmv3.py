# Copyright 2022 HuggingFace Inc.
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

from transformers.testing_utils import require_pytesseract, require_torch
from transformers.utils import is_pytesseract_available, is_torchvision_available

from ...test_image_processing_common import ImageProcessingTestMixin, prepare_image_inputs


if is_pytesseract_available():
    from transformers import LayoutLMv3ImageProcessor

    if is_torchvision_available():
        from transformers import LayoutLMv3ImageProcessorFast


class LayoutLMv3ImageProcessingTester:
    def __init__(
        self,
        parent,
        batch_size=7,
        num_channels=3,
        image_size=18,
        min_resolution=30,
        max_resolution=400,
        do_resize=True,
        size=None,
        apply_ocr=True,
    ):
        size = size if size is not None else {"height": 18, "width": 18}
        self.parent = parent
        self.batch_size = batch_size
        self.num_channels = num_channels
        self.image_size = image_size
        self.min_resolution = min_resolution
        self.max_resolution = max_resolution
        self.do_resize = do_resize
        self.size = size
        self.apply_ocr = apply_ocr

    def prepare_image_processor_dict(self):
        return {"do_resize": self.do_resize, "size": self.size, "apply_ocr": self.apply_ocr}

    def expected_output_image_shape(self, images):
        return self.num_channels, self.size["height"], self.size["width"]

    def prepare_image_inputs(self, equal_resolution=False, numpify=False, torchify=False):
        return prepare_image_inputs(
            batch_size=self.batch_size,
            num_channels=self.num_channels,
            min_resolution=self.min_resolution,
            max_resolution=self.max_resolution,
            equal_resolution=equal_resolution,
            numpify=numpify,
            torchify=torchify,
        )


@require_torch
@require_pytesseract
class LayoutLMv3ImageProcessingTest(ImageProcessingTestMixin, unittest.TestCase):
    image_processing_class = LayoutLMv3ImageProcessor if is_pytesseract_available() else None
    fast_image_processing_class = (
        LayoutLMv3ImageProcessorFast if (is_torchvision_available() and is_pytesseract_available()) else None
    )

    def setUp(self):
        super().setUp()
        self.image_processor_tester = LayoutLMv3ImageProcessingTester(self)

    @property
    def image_processor_dict(self):
        return self.image_processor_tester.prepare_image_processor_dict()

    def test_image_processor_properties(self):
        for image_processing_class in self.image_processor_list:
            image_processing = image_processing_class(**self.image_processor_dict)
            self.assertTrue(hasattr(image_processing, "do_resize"))
            self.assertTrue(hasattr(image_processing, "size"))
            self.assertTrue(hasattr(image_processing, "apply_ocr"))

    def test_image_processor_from_dict_with_kwargs(self):
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class.from_dict(self.image_processor_dict)
            self.assertEqual(image_processor.size, {"height": 18, "width": 18})

            image_processor = image_processing_class.from_dict(self.image_processor_dict, size=42)
            self.assertEqual(image_processor.size, {"height": 42, "width": 42})

    def test_LayoutLMv3_integration_test(self):
        from datasets import load_dataset

        ds = load_dataset("hf-internal-testing/fixtures_docvqa", split="test")

        # with apply_OCR = True
        for image_processing_class in self.image_processor_list:
            image_processor = image_processing_class()

            image = ds[0]["image"].convert("RGB")

            encoding = image_processor(image, return_tensors="pt")

            self.assertEqual(encoding.pixel_values.shape, (1, 3, 224, 224))
            self.assertEqual(len(encoding.words), len(encoding.boxes))

            # fmt: off
            # the words and boxes were obtained with Tesseract 5.3.0
            expected_words = [['11:14', 'to', '11:39', 'a.m', '11:39', 'to', '11:44', 'a.m.', '11:44', 'a.m.', 'to', '12:25', 'p.m.', '12:25', 'to', '12:58', 'p.m.', '12:58', 'to', '4:00', 'p.m.', '2:00', 'to', '5:00', 'p.m.', 'Coffee', 'Break', 'Coffee', 'will', 'be', 'served', 'for', 'men', 'and', 'women', 'in', 'the', 'lobby', 'adjacent', 'to', 'exhibit', 'area.', 'Please', 'move', 'into', 'exhibit', 'area.', '(Exhibits', 'Open)', 'TRRF', 'GENERAL', 'SESSION', '(PART', '|)', 'Presiding:', 'Lee', 'A.', 'Waller', 'TRRF', 'Vice', 'President', '“Introductory', 'Remarks”', 'Lee', 'A.', 'Waller,', 'TRRF', 'Vice', 'Presi-', 'dent', 'Individual', 'Interviews', 'with', 'TRRF', 'Public', 'Board', 'Members', 'and', 'Sci-', 'entific', 'Advisory', 'Council', 'Mem-', 'bers', 'Conducted', 'by', 'TRRF', 'Treasurer', 'Philip', 'G.', 'Kuehn', 'to', 'get', 'answers', 'which', 'the', 'public', 'refrigerated', 'warehousing', 'industry', 'is', 'looking', 'for.', 'Plus', 'questions', 'from', 'the', 'floor.', 'Dr.', 'Emil', 'M.', 'Mrak,', 'University', 'of', 'Cal-', 'ifornia,', 'Chairman,', 'TRRF', 'Board;', 'Sam', 'R.', 'Cecil,', 'University', 'of', 'Georgia', 'College', 'of', 'Agriculture;', 'Dr.', 'Stanley', 'Charm,', 'Tufts', 'University', 'School', 'of', 'Medicine;', 'Dr.', 'Robert', 'H.', 'Cotton,', 'ITT', 'Continental', 'Baking', 'Company;', 'Dr.', 'Owen', 'Fennema,', 'University', 'of', 'Wis-', 'consin;', 'Dr.', 'Robert', 'E.', 'Hardenburg,', 'USDA.', 'Questions', 'and', 'Answers', 'Exhibits', 'Open', 'Capt.', 'Jack', 'Stoney', 'Room', 'TRRF', 'Scientific', 'Advisory', 'Council', 'Meeting', 'Ballroom', 'Foyer']]  # noqa: E231
            # We get different outputs on CircleCI and on Github runners since 2025/06/26. It might be different versions of some 3rd party libraries in these 2 environments.
            expected_boxes_1 = [[[141, 57, 210, 69], [228, 58, 252, 69], [141, 75, 216, 88], [230, 79, 280, 88], [142, 260, 218, 273], [230, 261, 255, 273], [143, 279, 218, 290], [231, 282, 290, 291], [143, 342, 218, 354], [231, 345, 289, 355], [202, 362, 227, 373], [143, 379, 220, 392], [231, 382, 291, 394], [144, 714, 220, 726], [231, 715, 256, 726], [144, 732, 220, 745], [232, 736, 291, 747], [144, 769, 218, 782], [231, 770, 256, 782], [141, 788, 202, 801], [215, 791, 274, 804], [143, 826, 204, 838], [215, 826, 240, 838], [142, 844, 202, 857], [215, 847, 274, 859], [334, 57, 427, 69], [440, 57, 522, 69], [369, 75, 461, 88], [469, 75, 516, 88], [528, 76, 562, 88], [570, 76, 667, 88], [675, 75, 711, 87], [721, 79, 778, 88], [789, 75, 840, 88], [369, 97, 470, 107], [484, 94, 507, 106], [518, 94, 562, 107], [576, 94, 655, 110], [668, 94, 792, 109], [804, 95, 829, 107], [369, 113, 465, 125], [477, 116, 547, 125], [562, 113, 658, 125], [671, 116, 748, 125], [761, 113, 811, 125], [369, 131, 465, 143], [477, 133, 548, 143], [563, 130, 698, 145], [710, 130, 802, 146], [336, 171, 412, 183], [423, 171, 572, 183], [582, 170, 716, 184], [728, 171, 817, 187], [829, 171, 844, 186], [338, 197, 482, 212], [507, 196, 557, 209], [569, 196, 595, 208], [610, 196, 702, 209], [505, 214, 583, 226], [595, 214, 656, 227], [670, 215, 807, 227], [335, 259, 543, 274], [556, 259, 708, 272], [372, 279, 422, 291], [435, 279, 460, 291], [474, 279, 574, 292], [587, 278, 664, 291], [676, 278, 738, 291], [751, 279, 834, 291], [372, 298, 434, 310], [335, 341, 483, 354], [497, 341, 655, 354], [667, 341, 728, 354], [740, 341, 825, 354], [335, 360, 430, 372], [442, 360, 534, 372], [545, 359, 687, 372], [697, 360, 754, 372], [765, 360, 823, 373], [334, 378, 428, 391], [440, 378, 577, 394], [590, 378, 705, 391], [720, 378, 801, 391], [334, 397, 400, 409], [370, 416, 529, 429], [544, 416, 576, 432], [587, 416, 665, 428], [677, 416, 814, 429], [372, 435, 452, 450], [465, 434, 495, 447], [511, 434, 600, 447], [611, 436, 637, 447], [649, 436, 694, 451], [705, 438, 824, 447], [369, 453, 452, 466], [464, 454, 509, 466], [522, 453, 611, 469], [625, 453, 792, 469], [370, 472, 556, 488], [570, 472, 684, 487], [697, 472, 718, 485], [732, 472, 835, 488], [369, 490, 411, 503], [425, 490, 484, 503], [496, 490, 635, 506], [645, 490, 707, 503], [718, 491, 761, 503], [771, 490, 840, 503], [336, 510, 374, 521], [388, 510, 447, 522], [460, 510, 489, 521], [503, 510, 580, 522], [592, 509, 736, 525], [745, 509, 770, 522], [781, 509, 840, 522], [338, 528, 434, 541], [448, 528, 596, 541], [609, 527, 687, 540], [700, 528, 792, 541], [336, 546, 397, 559], [407, 546, 431, 559], [443, 546, 525, 560], [537, 546, 680, 562], [695, 546, 714, 559], [722, 546, 837, 562], [336, 565, 449, 581], [461, 565, 485, 577], [497, 565, 665, 581], [681, 565, 718, 577], [732, 565, 837, 580], [337, 584, 438, 597], [452, 583, 521, 596], [535, 584, 677, 599], [690, 583, 787, 596], [801, 583, 825, 596], [338, 602, 478, 615], [492, 602, 530, 614], [543, 602, 638, 615], [650, 602, 676, 614], [688, 602, 788, 615], [802, 602, 843, 614], [337, 621, 502, 633], [516, 621, 615, 637], [629, 621, 774, 636], [789, 621, 827, 633], [337, 639, 418, 652], [432, 640, 571, 653], [587, 639, 731, 655], [743, 639, 769, 652], [780, 639, 841, 652], [338, 658, 440, 673], [455, 658, 491, 670], [508, 658, 602, 671], [616, 658, 638, 670], [654, 658, 835, 674], [337, 677, 429, 689], [337, 714, 482, 726], [495, 714, 548, 726], [561, 714, 683, 726], [338, 770, 461, 782], [474, 769, 554, 785], [489, 788, 562, 803], [576, 788, 643, 801], [656, 787, 751, 804], [764, 788, 844, 801], [334, 825, 421, 838], [430, 824, 574, 838], [584, 824, 723, 841], [335, 844, 450, 857], [464, 843, 583, 860], [628, 862, 755, 875], [769, 861, 848, 878]]]  # noqa: E231
            expected_boxes_2 = [[[141, 57, 214, 69], [228, 58, 252, 69], [141, 75, 216, 88], [230, 79, 280, 88], [142, 260, 218, 273], [230, 261, 255, 273], [143, 279, 218, 290], [231, 282, 290, 291], [143, 342, 218, 354], [231, 345, 289, 355], [202, 362, 227, 373], [143, 379, 220, 392], [231, 382, 291, 394], [144, 714, 220, 726], [231, 715, 256, 726], [144, 732, 220, 745], [232, 736, 291, 747], [144, 769, 218, 782], [231, 770, 256, 782], [141, 788, 202, 801], [215, 791, 274, 804], [143, 826, 204, 838], [215, 826, 240, 838], [142, 844, 202, 857], [215, 847, 274, 859], [334, 57, 427, 69], [440, 57, 522, 69], [369, 75, 461, 88], [469, 75, 516, 88], [528, 76, 562, 88], [570, 76, 667, 88], [675, 75, 711, 87], [721, 79, 778, 88], [789, 75, 840, 88], [369, 97, 470, 107], [484, 94, 507, 106], [518, 94, 562, 107], [576, 94, 655, 110], [668, 94, 792, 109], [804, 95, 829, 107], [369, 113, 465, 125], [477, 116, 547, 125], [562, 113, 658, 125], [671, 116, 748, 125], [761, 113, 811, 125], [369, 131, 465, 143], [477, 133, 548, 143], [563, 130, 698, 145], [710, 130, 802, 146], [336, 171, 412, 183], [423, 171, 572, 183], [582, 170, 716, 184], [728, 171, 817, 187], [829, 171, 844, 186], [338, 197, 482, 212], [507, 196, 557, 209], [569, 196, 595, 208], [610, 196, 702, 209], [505, 214, 583, 226], [595, 214, 656, 227], [670, 215, 807, 227], [335, 259, 543, 274], [556, 259, 708, 272], [372, 279, 422, 291], [435, 279, 460, 291], [474, 279, 574, 292], [587, 278, 664, 291], [676, 278, 738, 291], [751, 279, 834, 291], [372, 298, 434, 310], [335, 341, 483, 354], [497, 341, 655, 354], [667, 341, 728, 354], [740, 341, 825, 354], [335, 360, 430, 372], [442, 360, 534, 372], [545, 359, 687, 372], [697, 360, 754, 372], [765, 360, 823, 373], [334, 378, 428, 391], [440, 378, 577, 394], [590, 378, 705, 391], [720, 378, 801, 391], [334, 397, 400, 409], [370, 416, 529, 429], [544, 416, 576, 432], [587, 416, 665, 428], [677, 416, 814, 429], [372, 435, 452, 450], [465, 434, 495, 447], [511, 434, 600, 447], [611, 436, 637, 447], [649, 436, 694, 451], [705, 438, 824, 447], [369, 453, 452, 466], [464, 454, 509, 466], [522, 453, 611, 469], [625, 453, 792, 469], [370, 472, 556, 488], [570, 472, 684, 487], [697, 472, 718, 485], [732, 472, 835, 488], [369, 490, 411, 503], [425, 490, 484, 503], [496, 490, 635, 506], [645, 490, 707, 503], [718, 491, 761, 503], [771, 490, 840, 503], [336, 510, 374, 521], [388, 510, 447, 522], [460, 510, 489, 521], [503, 510, 580, 522], [592, 509, 736, 525], [745, 509, 770, 522], [781, 509, 840, 522], [338, 528, 434, 541], [448, 528, 596, 541], [609, 527, 687, 540], [700, 528, 792, 541], [336, 546, 397, 559], [407, 546, 431, 559], [443, 546, 525, 560], [537, 546, 680, 562], [688, 546, 714, 559], [722, 546, 837, 562], [336, 565, 449, 581], [461, 565, 485, 577], [497, 565, 665, 581], [681, 565, 718, 577], [732, 565, 837, 580], [337, 584, 438, 597], [452, 583, 521, 596], [535, 584, 677, 599], [690, 583, 787, 596], [801, 583, 825, 596], [338, 602, 478, 615], [492, 602, 530, 614], [543, 602, 638, 615], [650, 602, 676, 614], [688, 602, 788, 615], [802, 602, 843, 614], [337, 621, 502, 633], [516, 621, 615, 637], [629, 621, 774, 636], [789, 621, 827, 633], [337, 639, 418, 652], [432, 640, 571, 653], [587, 639, 731, 655], [743, 639, 769, 652], [780, 639, 841, 652], [338, 658, 440, 673], [455, 658, 491, 670], [508, 658, 602, 671], [616, 658, 638, 670], [654, 658, 835, 674], [337, 677, 429, 689], [337, 714, 482, 726], [495, 714, 548, 726], [561, 714, 683, 726], [338, 770, 461, 782], [474, 769, 554, 785], [489, 788, 562, 803], [576, 788, 643, 801], [656, 787, 751, 804], [764, 788, 844, 801], [334, 825, 421, 838], [430, 824, 574, 838], [584, 824, 723, 841], [335, 844, 450, 857], [464, 843, 583, 860], [628, 862, 755, 875], [769, 861, 848, 878]]]  # noqa: E231
            # fmt: on

            self.assertListEqual(encoding.words, expected_words)
            self.assertIn(encoding.boxes, [expected_boxes_1, expected_boxes_2])

            # with apply_OCR = False
            image_processor = image_processing_class(apply_ocr=False)

            encoding = image_processor(image, return_tensors="pt")

            self.assertEqual(encoding.pixel_values.shape, (1, 3, 224, 224))
