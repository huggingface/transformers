import gradio as gr
from helpers import clean_model_id
from checks_generate import check_generate
from checks_ci import check_ci
from checks_attention import check_attention_support


# ── orchestrator ─────────────────────────────────────────────────────────────

def run_checks(model_id_raw: str):
    if not model_id_raw.strip():
        msg = "Please enter a model ID or URL."
        return msg, "", msg, msg, ""
    model_id = clean_model_id(model_id_raw)
    gen_summary, gen_details = check_generate(model_id)
    ci = check_ci(model_id)
    attn_summary, attn_details = check_attention_support(model_id)
    return gen_summary, gen_details, ci, attn_summary, attn_details


# ── UI ────────────────────────────────────────────────────────────────────────

EXAMPLES = [
    ["Qwen/Qwen2.5-7B-Instruct"],
    ["google/gemma-2-2b-it"],
    ["meta-llama/Llama-3.2-1B"],
    ["mistralai/Ministral-8B-Instruct-2410"],
]

with gr.Blocks(title="vibecheck") as demo:
    gr.Markdown(
        "# ✅ vibecheck\n"
        "Paste a Hugging Face checkpoint to check config/tokenizer/generation, "
        "CI test status, and attention implementation support."
    )

    with gr.Row():
        model_input = gr.Textbox(
            label="Model ID or URL",
            placeholder="Qwen/Qwen2.5-7B-Instruct  or  https://huggingface.co/...",
            scale=5,
        )
        run_btn = gr.Button("Run Checks", variant="primary", scale=1, min_width=130)

    with gr.Row():
        # Column 1 — generate
        with gr.Column():
            generate_summary = gr.Markdown(label="Generate Check")
            with gr.Accordion("Details", open=False):
                generate_details = gr.Markdown()

        # Column 2 — CI
        with gr.Column():
            ci_out = gr.Markdown(label="CI Status")

        # Column 3 — attention
        with gr.Column():
            attn_summary = gr.Markdown(label="Attention Support")
            with gr.Accordion("Details", open=False):
                attn_details = gr.Markdown()

    run_btn.click(
        fn=run_checks,
        inputs=[model_input],
        outputs=[generate_summary, generate_details, ci_out, attn_summary, attn_details],
    )

    gr.Examples(examples=EXAMPLES, inputs=[model_input])


if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
