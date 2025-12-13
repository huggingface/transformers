from transformers import Qwen2_5_VLForConditionalGeneration


def main():
    """
    Reproduce with a real quantized model: load the W8A8-quantized Qwen2.5-VL model.
    Previously, int8 weights caused a dtype error during initialization when normal_ was applied.
    Ensure China mainland mirror is set via HF_ENDPOINT before running to speed up downloads.
    """

    model_id = "RedHatAI/Qwen2.5-VL-7B-Instruct-quantized.w8a8"
    _ = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype="auto",
        trust_remote_code=True,
    )


if __name__ == "__main__":
    main()
