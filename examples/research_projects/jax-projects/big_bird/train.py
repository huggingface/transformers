import os
from dataclasses import replace

import jax
import wandb
from bigbird_flax import Args, DataCollator, FlaxBigBirdForNaturalQuestions, Trainer, build_tx, train_step, val_step
from datasets import load_dataset
from flax import jax_utils

from transformers import BigBirdTokenizerFast


if __name__ == "__main__":
    print("#################### AVAILABLE DEVICES ####################")
    print(jax.devices())
    print("###########################################################")

    # setup for wandb sweep
    args = Args()
    logger = wandb.init(project="bigbird-natural-questions", config=args.__dict__)
    wandb_args = dict(logger.config)
    del wandb_args["batch_size"]
    args = replace(args, **wandb_args)
    base_dir = args.base_dir + "-" + wandb.run.id
    args = replace(args, base_dir=base_dir)
    print(args)

    tr_dataset = load_dataset("json", data_files=args.tr_data_path)["train"]
    val_dataset = load_dataset("json", data_files=args.val_data_path)["train"]

    # drop extra batch for now
    indices = range(len(tr_dataset) - len(tr_dataset) % args.batch_size)
    tr_dataset = tr_dataset.shuffle().select(indices)
    indices = range(len(val_dataset) - len(val_dataset) % args.batch_size)
    val_dataset = val_dataset.shuffle().select(indices)

    if os.environ.get("TRAIN_ON_SMALL", "false") == "true":
        tr_dataset = tr_dataset.shuffle().select(range(80000))
        val_dataset = val_dataset.shuffle().select(range(8000))

    print(tr_dataset)
    print(val_dataset)

    model = FlaxBigBirdForNaturalQuestions.from_pretrained(
        args.model_id, block_size=args.block_size, num_random_blocks=args.num_random_blocks
    )
    tokenizer = BigBirdTokenizerFast.from_pretrained(args.model_id)
    data_collator = DataCollator(pad_id=tokenizer.pad_token_id, max_length=4096)

    tx_args = {
        "lr": args.lr,
        "init_lr": args.init_lr,
        "warmup_steps": args.warmup_steps,
        "num_train_steps": args.max_epochs * (len(tr_dataset) // args.batch_size),
        "weight_decay": args.weight_decay,
    }
    tx, lr = build_tx(**tx_args)

    trainer = Trainer(
        args=args,
        data_collator=data_collator,
        model_save_fn=model.save_pretrained,
        train_step_fn=train_step,
        val_step_fn=val_step,
        logger=logger,
        scheduler_fn=lr,
    )

    ckpt_dir = None
    state = trainer.create_state(model, tx, num_train_steps=tx_args["num_train_steps"], ckpt_dir=ckpt_dir)
    try:
        trainer.train(state, tr_dataset, val_dataset)
    except KeyboardInterrupt:
        print("Oooops; TRAINING STOPPED UNFORTUNATELY")

    print("SAVING WEIGHTS IN `final-weights`")
    params = jax_utils.unreplicate(state.params)
    model.save_pretrained(os.path.join(args.base_dir, "final-weights"), params=params)
