import gc

import requests
from datasets import Dataset

from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    LlavaForConditionalGeneration,
    Trainer,
    TrainingArguments,
    is_torch_available,
    is_vision_available,
)
from transformers.testing_utils import TestCasePlus, require_bitsandbytes, require_peft, require_torch, slow


if is_vision_available():
    from PIL import Image

if is_torch_available():
    import torch
else:
    is_torch_greater_or_equal_than_2_0 = False


# Integration test for confirming autocast with trainer and accelerate works
# correctly. Confirms type error found
# https://github.com/huggingface/transformers/pull/29721 in is fixed
@require_torch
class LlavaForConditionalGenerationIntegrationTest(TestCasePlus):
    def setUp(self):
        super().setUp()
        self.processor = AutoProcessor.from_pretrained(
            "llava-hf/bakLlava-v1-hf", padding_side="left", truncation_side="right"
        )

    def tearDown(self):
        gc.collect()
        torch.cuda.empty_cache()

    @slow
    @require_bitsandbytes
    @require_peft
    def test_model_trainer_integration_test(self):
        from peft import LoraConfig, PeftModelForCausalLM

        def image_prompt_generator():
            prompts = [
                "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
                "USER: <image>\nWhat is this?\nASSISTANT:",
            ]
            image_urls = [
                "https://llava-vl.github.io/static/images/view.jpg",
                "http://images.cocodataset.org/val2017/000000039769.jpg",
            ]

            for prompt, image_url in zip(prompts, image_urls):
                image = Image.open(requests.get(image_url, stream=True).raw)
                yield {"image": image, "prompt": prompt}

        def process_image_prompt(data):
            processed = self.processor(
                data["prompt"], images=data["image"], return_tensors="pt", padding=True, max_length=512
            )
            return {
                "input_ids": processed["input_ids"].squeeze(),
                "attention_mask": processed["attention_mask"].squeeze(),
                "pixel_values": processed["pixel_values"].squeeze(),
            }

        train_dataset = Dataset.from_generator(image_prompt_generator).map(process_image_prompt)
        bits_and_bytes_config = BitsAndBytesConfig(
            load_in_4bit=True,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            "llava-hf/bakLlava-v1-hf", quantization_config=bits_and_bytes_config
        )
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="CAUSAL_LM",
            lora_dropout=0.0,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        model = PeftModelForCausalLM(model, peft_config, adapter_name="lora_default")
        data_collator = DataCollatorForLanguageModeling(self.processor.tokenizer, mlm=False)

        output_dir = self.get_auto_remove_tmp_dir()
        trainer = Trainer(
            model=model,
            train_dataset=train_dataset,
            tokenizer=self.processor.tokenizer,
            args=TrainingArguments(output_dir, fp16=True, learning_rate=2e-5, num_train_epochs=1),
            data_collator=data_collator,
        )
        trainer.train()

        prompts = [
            "USER: <image>\nWhat are the things I should be cautious about when I visit this place? What should I bring with me?\nASSISTANT:",
            "USER: <image>\nWhat is this?\nASSISTANT:",
        ]
        image1 = Image.open(requests.get("https://llava-vl.github.io/static/images/view.jpg", stream=True).raw)
        image2 = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

        inputs = self.processor(prompts, images=[image1, image2], return_tensors="pt", padding=True)

        output = model(**inputs)
        expected_slice = torch.tensor(
            [[-3.5664, -3.5625, -0.4309], [-5.8242, -5.6914, -1.3242], [-5.4805, -5.9375, 1.1465]],
            dtype=torch.float32,
        )

        assert torch.allclose(output["logits"][0, :3, :3], expected_slice, atol=1e-3)
