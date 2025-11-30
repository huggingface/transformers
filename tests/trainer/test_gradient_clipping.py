from datasets import Dataset

from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments


def test_skip_unnecessary_grad_clip(monkeypatch):
    # Dummy model and data
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    data = {"text": ["hello world", "foo bar"], "label": [0, 1]}
    ds = Dataset.from_dict(data)
    ds = ds.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

    args = TrainingArguments(
        output_dir="./test_output",
        skip_unnecessary_grad_clip=True,  # <-- YOUR FEATURE!
        max_grad_norm=1e8,  # <-- Set threshold extremely high so grad norm is always below!
        per_device_train_batch_size=2,
        num_train_epochs=1,
    )

    def fake_clip_grad_norm(*args, **kwargs):
        raise RuntimeError("Should not clip! Grad norm is under threshold.")

    trainer = Trainer(model=model, args=args, train_dataset=ds)
    monkeypatch.setattr(trainer.accelerator, "clip_grad_norm_", fake_clip_grad_norm)

    # Run one training step and make sure no runtime error raised (no clipping triggered)
    trainer.train()

    # Check logged grad_norm value is less than threshold (since threshold is huge)
    logged_norms = [entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry]
    assert all(norm < args.max_grad_norm for norm in logged_norms), f"Grad norm logging failed! {logged_norms}"
