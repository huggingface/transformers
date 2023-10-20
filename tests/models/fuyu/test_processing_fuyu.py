import io
import unittest

import requests

from transformers import AutoTokenizer, is_torch_available, is_vision_available
from transformers.testing_utils import require_torch, require_torch_gpu, slow


if is_vision_available():
    from PIL import Image

if is_vision_available() and is_torch_available():
    from transformers import FuyuImageProcessor, FuyuProcessor

if is_torch_available():
    import torch

    from transformers.models.fuyu.processing_fuyu import construct_full_unpacked_stream, full_unpacked_stream_to_tensor


@require_torch
@require_torch_gpu
@slow
class FuyuProcessingTest(unittest.TestCase):  # TODO Which mixins do we add here?
    """ """

    def setUp(self):
        pretrained_model_name = "huggingface/pre_release_model"
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        image_processor = FuyuImageProcessor()

        processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
        text_prompt = "Generate a coco-style caption.\\n"
        bus_image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/bus.png"
        bus_image_pil = Image.open(io.BytesIO(requests.get(bus_image_url).content))

        self.one_image_bus_model_inputs = processor(text=text_prompt, images=bus_image_pil)

    def test_fuyu_processing(self):
        """
        Test to ensure that the standard processing on a gold example matches adept's code.
        """
        # fmt: off
        EXPECTED_IMAGE_PATCH_INPUTS = torch.Tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, -1, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, -1, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, -1, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, -1, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, -1, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, -1, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, -1, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175, -1, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194, 195, 196, 197, -1, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, -1, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, -1, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 258, 259, 260, 261, 262, 263, -1, 264, 265, 266, 267, 268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, -1, 286, 287, 288, 289, 290, 291, 292, 293, 294, 295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,]]).to(torch.int64)
        EXPECTED_PADDED_UNPACKED_TOKEN_INPUTS = torch.Tensor([[71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71011, 71019, 1, 128340, 71374, 71389, 120412, 71377, 71835, 71374, 73615, 71375, 71399, 71435, 71122,]]).to(torch.int64)
        # fmt: on
        torch.testing.assert_close(
            self.one_image_bus_model_inputs["image_patches_indices"], EXPECTED_IMAGE_PATCH_INPUTS
        )
        torch.testing.assert_close(self.one_image_bus_model_inputs["input_ids"], EXPECTED_PADDED_UNPACKED_TOKEN_INPUTS)


@require_torch
class TestImageTextProcessingUtils(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.new_seq_len = 8
        self.num_sub_sequences = 1

        self.all_bi_tokens_to_place = [4, 6]
        self.full_unpacked_stream = [torch.tensor([1, 2, 3, 4]), torch.tensor([5, 6, 7, 8, 9, 10])]
        self.fill_value = 0

        self.num_real_text_tokens = [[3, 2], [2, 4]]
        # Here the input stream is padded to avoid inconsistencies (current model release matches)
        self.input_stream = torch.tensor([[[1, 2, 3], [4, 5, 0]], [[6, 7, 0], [8, 9, 10]]])
        self.image_tokens = [
            [torch.tensor([1, 2]), torch.tensor([3])],
            [torch.tensor([4, 5, 6]), torch.tensor([7, 8])],
        ]

    def test_full_unpacked_stream_to_tensor(self):
        result = full_unpacked_stream_to_tensor(
            self.all_bi_tokens_to_place,
            self.full_unpacked_stream,
            self.fill_value,
            self.batch_size,
            self.new_seq_len,
            offset=0,
        )
        EXPECTED_TENSOR = torch.tensor([[1, 2, 3, 4, 0, 0, 0, 0], [5, 6, 7, 8, 9, 10, 0, 0]])
        self.assertTrue(torch.equal(result, EXPECTED_TENSOR))

    def test_construct_full_unpacked_stream(self):
        result = construct_full_unpacked_stream(
            self.num_real_text_tokens, self.input_stream, self.image_tokens, self.batch_size, self.num_sub_sequences
        )
        EXPECTED_UNPACKED_STREAM = [torch.tensor([1, 2, 1, 2, 3]), torch.tensor([4, 5, 6, 6, 7])]
        for i in range(len(result)):
            self.assertTrue(torch.equal(result[i], EXPECTED_UNPACKED_STREAM[i]))


@require_torch
class TestProcessImagesForModelInput(unittest.TestCase):
    def setUp(self):
        """
        Adding a mix of present and absent images.
        """
        self.image_processor = FuyuImageProcessor()

        self.image_input = torch.randn([1, 1, 3, 64, 64])
        self.image_present = torch.tensor([[1]])
        self.image_unpadded_h = torch.tensor([[45]])  # Adjusted for subsequence of 1
        self.image_unpadded_w = torch.tensor([[50]])  # Adjusted for subsequence of 1
        self.image_patch_dim_h = 16
        self.image_patch_dim_w = 16
        self.image_placeholder_id = 999
        self.image_newline_id = 888
        self.variable_sized = True

    def test_process_images_for_model_input_fixed_sized(self):
        self.variable_sized = False
        result = self.image_processor.process_images_for_model_input(
            image_input=self.image_input,
            image_present=self.image_present,
            image_unpadded_h=self.image_unpadded_h,
            image_unpadded_w=self.image_unpadded_w,
            image_patch_dim_h=self.image_patch_dim_h,
            image_patch_dim_w=self.image_patch_dim_w,
            image_placeholder_id=self.image_placeholder_id,
            image_newline_id=self.image_newline_id,
            variable_sized=self.variable_sized,
        )
        print(result["images"][0][0])
        self.assertEqual(result["images"][0][0].shape, torch.Size([3, 64, 64]))
