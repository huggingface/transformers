from transformers import AutoConfig, AutoTokenizer
from huggingface_hub import InferenceClient
from helpers import HF_TOKEN

__all__ = ["check_generate"]

DEFAULT_PROMPT = "Hello, my name is"
DEFAULT_MAX_TOKENS = 30


def check_generate(model_id: str) -> tuple[str, str]:
    summary, details = [], []

    try:
        config = AutoConfig.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
        arch = (config.architectures or ["unknown"])[0]
        is_generative = any(x in arch for x in ("CausalLM", "ConditionalGeneration", "LMHead", "ForSeq2Seq"))
        details.append(f"**Config:** `{config.model_type}` / `{arch}`")
    except Exception as e:
        return f"❌ Config: `{e}`", ""

    try:
        tok = AutoTokenizer.from_pretrained(model_id, token=HF_TOKEN, trust_remote_code=True)
        tok_name = type(tok).__name__
        summary.append(f"✅ `{tok_name}`")
        details.append(f"**Tokenizer:** `{tok_name}` (vocab: {tok.vocab_size:,})")
    except Exception as e:
        summary.append("❌ Tokenizer")
        details.append(f"❌ **Tokenizer:** `{e}`")

    if not is_generative:
        summary.append(f"⚠️ `{arch}` (no generate)")
        details.append(f"**Generation:** skipped — not a generative architecture")
        return " &nbsp; ".join(summary), "\n\n".join(details)

    client = InferenceClient(model=model_id, token=HF_TOKEN)
    try:
        output = client.text_generation(DEFAULT_PROMPT, max_new_tokens=DEFAULT_MAX_TOKENS)
        summary.append(f"✅ `{arch}`")
        details.append(f"**Generation** (text-generation):\n- Prompt: _{DEFAULT_PROMPT}_\n- Output: _{output.strip()}_")
    except Exception as e:
        if "conversational" in str(e).lower() or "not supported for task" in str(e).lower():
            try:
                resp = client.chat_completion(
                    messages=[{"role": "user", "content": DEFAULT_PROMPT}],
                    max_tokens=DEFAULT_MAX_TOKENS,
                )
                output = resp.choices[0].message.content
                summary.append(f"✅ `{arch}`")
                details.append(f"**Generation** (chat_completion):\n- Prompt: _{DEFAULT_PROMPT}_\n- Output: _{output.strip()}_")
            except Exception as e2:
                summary.append("❌ Generation")
                details.append(f"❌ **Generation:**\n- text-generation: `{e}`\n- chat_completion: `{e2}`")
        else:
            summary.append("❌ Generation")
            details.append(f"❌ **Generation:** `{e}`")

    return " &nbsp; ".join(summary), "\n\n".join(details)
