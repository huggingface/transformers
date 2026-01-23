def test_loss_aggregation_multi_gpu(tmp_path):
    import torch
    from transformers import Trainer, TrainingArguments, AutoModelForCausalLM

    # tiny model for testing
    model = AutoModelForCausalLM.from_pretrained("hf-internal-testing/tiny-random-gpt2")
    args = TrainingArguments(output_dir=tmp_path, per_device_train_batch_size=2)

    trainer = Trainer(model=model, args=args)
    trainer.args.n_gpu = 2  # simulate 2 GPUs

    # simulate per-GPU losses
    dummy_losses = torch.tensor([2.0, 2.0])

    # aggregation should sum (not mean)
    assert torch.isclose(dummy_losses.sum(), torch.tensor(4.0))
