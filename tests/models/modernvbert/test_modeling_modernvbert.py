# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from transformers import is_torch_available
from transformers.testing_utils import require_torch, slow


if is_torch_available():
    import torch
    from huggingface_hub import hf_hub_download
    from PIL import Image

    from transformers import AutoProcessor, AutoTokenizer
    from transformers.models.modernvbert.modeling_modernvbert import ModernVBertForMaskedLM


@require_torch
class ModernVBertIntegrationTest(unittest.TestCase):
    @slow
    def test_masked_lm_prediction(self):
        model_id = "ModernVBERT/modernvbert"

        processor = AutoProcessor.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = ModernVBertForMaskedLM.from_pretrained(
            model_id,
            torch_dtype=torch.float32,  # use torch_dtype=torch.bfloat16 for flash attention
            # _attn_implementation="flash_attention_2",
            trust_remote_code=True,
        )

        image = Image.open(hf_hub_download("HuggingFaceTB/SmolVLM", "example_images/rococo.jpg", repo_type="space"))
        text = "This [MASK] is on the wall."

        # Create input messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": text},
                ],
            },
        ]

        # Prepare inputs
        prompt = processor.apply_chat_template(messages, add_generation_prompt=False)
        inputs = processor(text=prompt, images=[image], return_tensors="pt")

        # Inference
        with torch.no_grad():
            outputs = model(**inputs)

        # To get predictions for the mask:
        masked_index = inputs["input_ids"][0].tolist().index(tokenizer.mask_token_id)

        # assert that top k is values=tensor([21.2242, 21.1116, 19.1809, 18.4607, 17.9964]), indices=tensor([13497,  5406,  2460,  7512,  3665])
        self.assertEqual(
            [round(x, 4) for x in outputs.logits[0, masked_index].topk(5).values.tolist()],
            [21.2242, 21.1116, 19.1809, 18.4607, 17.9964],
        )
        self.assertEqual(
            outputs.logits[0, masked_index].topk(5).indices.tolist(),
            [13497, 5406, 2460, 7512, 3665],
        )