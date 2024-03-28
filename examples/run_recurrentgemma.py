"""Temporary example to test whether model generation works correctly."""

from transformers import AutoTokenizer
from transformers import RecurrentGemmaForCausalLM
from transformers.models import recurrentgemma


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b-it")
    model = RecurrentGemmaForCausalLM(
        config=recurrentgemma.RecurrentGemmaConfig(
            num_hidden_layers=3,
            vocab_size=256000,
            width=256,
            mlp_expanded_width=3 * 256,
            num_heads=2,
            lru_width=None,
            embeddings_scale_by_sqrt_dim=True,
            attention_window_size=64,
            conv1d_width=4,
            logits_soft_cap=30.0,
            rms_norm_eps=1e-6,
        )
    )

    input_text = "Write me a poem about Machine Learning."
    model_kwargs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**model_kwargs)
    print(tokenizer.decode(outputs[0]))


if __name__ == "__main__":
    main()
