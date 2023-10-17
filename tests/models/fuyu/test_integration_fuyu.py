from transformers import AutoTokenizer
from transformers import FuyuConfig, FuyuModel
from transformers.models.fuyu.processing_fuyu import FuyuProcessor
from transformers.models.fuyu.image_processing_fuyu import FuyuImageProcessor
from PIL import Image
import torch

import unittest

from transformers.testing_utils import (
    require_torch_gpu,
    slow,
)


@require_torch_gpu
@slow
class FuyuIntegrationTest(unittest.TestCase):
    def setUp(self):
        pretrained_model_name = 'huggingface/pre_release_model'
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        image_processor = FuyuImageProcessor()

        processor = FuyuProcessor(image_processor=image_processor, tokenizer=tokenizer)
        text_prompt = "Generate a coco-style caption.\\n"

        image_path = "/fsx/pablo/adept-collab/adept-mm/mm-inference-for-hf/bus.png"
        image_pil = Image.open(image_path)

        self.model_inputs = processor(text=text_prompt, images=[image_pil])

        self.model_config = FuyuConfig()
        self.model = FuyuModel(self.model_config).from_pretrained(pretrained_model_name)

    def test_fuyu_processing(self):
        """
        Test to ensure that the standard processing on a gold example matches adept's code.
        """
        assert torch.equal(self.model_inputs["image_patch_input_indices"], torch.Tensor([[
            0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
            14,  15,  16,  17,  18,  19,  20,  21,  -1,  22,  23,  24,  25,  26,
            27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,  39,  40,
            41,  42,  43,  -1,  44,  45,  46,  47,  48,  49,  50,  51,  52,  53,
            54,  55,  56,  57,  58,  59,  60,  61,  62,  63,  64,  65,  -1,  66,
            67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,  80,
            81,  82,  83,  84,  85,  86,  87,  -1,  88,  89,  90,  91,  92,  93,
            94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107,
            108, 109,  -1, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131,  -1, 132, 133,
            134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147,
            148, 149, 150, 151, 152, 153,  -1, 154, 155, 156, 157, 158, 159, 160,
            161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174,
            175,  -1, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187,
            188, 189, 190, 191, 192, 193, 194, 195, 196, 197,  -1, 198, 199, 200,
            201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213, 214,
            215, 216, 217, 218, 219,  -1, 220, 221, 222, 223, 224, 225, 226, 227,
            228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241,
            -1, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254,
            255, 256, 257, 258, 259, 260, 261, 262, 263,  -1, 264, 265, 266, 267,
            268, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281,
            282, 283, 284, 285,  -1, 286, 287, 288, 289, 290, 291, 292, 293, 294,
            295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
            -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1]]))

        assert torch.equal(self.model_inputs['image_padded_unpacked_tokens_tensor'], torch.Tensor([[
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71019,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71019,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71019,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71019,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71019,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71019,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71019,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71019,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71019,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71019,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71019,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71019,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,  71011,
            71011,  71011,  71011,  71011,  71011,  71011,  71019,      1, 128340,
            71374,  71389, 120412,  71377,  71835,  71374,  73615,  71375,  71399,
            71435,  71122,  71013,  71013,  71013,  71013,  71013,  71013,  71013,
            71013,  71013,  71013]]))

    def test_model_embeddings_match_adept(self):
        word_embeddings = self.model.embed_tokens(self.model_inputs['image_padded_unpacked_tokens_tensor'][0][None, :])
        expected_word_embedding_start = torch.Tensor([2.8908e-06, -1.4961e-05, -2.9564e-05, -1.4901e-05, -2.5153e-05,
                                                      -1.9312e-05, -2.5511e-05,  1.9431e-05, -7.5698e-06, -3.5286e-05])
        expected_word_embedding_end = torch.Tensor([8.8811e-06, -3.7909e-05, -4.8637e-05,  2.1100e-05, -1.4544e-05,
                                                    -3.0756e-05,  4.3511e-06, -1.5080e-05,  2.5153e-05])
        assert torch.allclose(word_embeddings[0][0][0:10], expected_word_embedding_start, atol=1e-7)
        assert torch.allclose(word_embeddings[0][0][-9:], expected_word_embedding_end, atol=1e-7)
        assert word_embeddings.shape == torch.Size([1, 335, 4096])

        continuous_embeddings = self.model.vision_embed_tokens(self.model_inputs['image_patches'][0][0]).unsqueeze(0)

        expected_continuous_embedding_start = torch.Tensor([-0.2891,  0.0649, -0.0175,  0.1641, -0.4844,
                                                            -0.9062,  0.4473,  0.2412, -0.2461, -0.0430])
        expected_continuous_embedding_end = torch.Tensor([-0.2754, -0.1836,  0.2422, -0.3711,  0.0564,
                                                          -0.1099,  0.0378,  0.1367, -0.2100])
        assert torch.allclose(continuous_embeddings[0][0][0:10], expected_continuous_embedding_start, atol=1e-4)
        assert torch.allclose(continuous_embeddings[0][0][-9:], expected_continuous_embedding_end, atol=1e-4)

        assert continuous_embeddings[0].shape == torch.Size([308, 4096])
