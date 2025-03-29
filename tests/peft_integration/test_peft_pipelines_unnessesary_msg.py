from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from transformers.testing_utils import (
    require_peft,
    require_torch,
    slow,
)
from transformers.utils import is_torch_available


@slow
@require_torch
@require_peft
def test_peft_pipelines_unnessesary_msg():
    if is_torch_available():
        import torch

    from peft import PeftModel

    ADAPTER_PATH = "sajjadhadi/Disease-Diagnosis-Qwen2.5-0.5B"
    BASE_PATH = "Qwen/Qwen2.5-0.5B"

    # input
    text = "Who is a Elon Musk?"

    model = AutoModelForCausalLM.from_pretrained(
        BASE_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(BASE_PATH)
    default_generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        max_length=10,
        truncation=True,
        max_new_tokens=10,  # 2. 후행 공백 제거
    )
    print(f"this is base model result: {default_generator(text)}")

    lora_model = PeftModel.from_pretrained(
        model,
        ADAPTER_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    lora_generator = pipeline(
        task="text-generation",
        model=lora_model,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype=torch.float16,
        max_length=10,
        truncation=True,
        max_new_tokens=10,
    )
    print(f"this is lora model result: {lora_generator(text)}")

    print("\nIf no error message appears between the two outputs, the fix is working correctly!")
