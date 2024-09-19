import os
import click
import numpy as np
from pathlib import Path
from datasets import load_dataset
import evaluate
from transformers import ViTImageProcessor, ViTForImageClassification
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from torch.utils.data import DataLoader
from timm.utils import accuracy, AverageMeter

def get_dataloaders(dataset, batch_size):
    train_dataset = load_dataset(
        "imagefolder", data_dir=os.path.join(dataset, "train"), split="train"
    )
    val_dataset = load_dataset(
        "imagefolder", data_dir=os.path.join(dataset, "val"), split="validation"
    )

    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def transforms(batch):
        # Some of images in ImageNet turned out to be of one channel (gray).
        images = [img.convert("RGB") for img in batch["image"]]
        batch["image"] = image_processor(images, return_tensors="pt")["pixel_values"]
        batch["label"] = torch.tensor(batch["label"])
        return batch

    train_dataset.set_transform(transforms)
    val_dataset.set_transform(transforms)

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    return train_dataloader, val_dataloader


@click.command()
@click.option(
    "--dataset",
    type=click.Path(path_type=Path),
    help="Path to dataset.",
)
@click.option(
    "--batch",
    required=True,
    type=int,
    help="Batch size.",
)
@click.option(
    "--epoch",
    default=400,
    type=int,
    help="number of epochs.",
)
@click.option(
    "--lr",
    default=5e-3,
    type=float,
    help="Initial learning rate.",
)
@click.option(
    "--save",
    type=click.Path(path_type=Path),
    default="outputs", 
    help="Path to save a checkpoint for the student model.",
)
def main(dataset, batch, epoch, lr, save):
    train_dataloader, val_dataloader = get_dataloaders(dataset, batch)

    teacher_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224"
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # placeholder
    student_model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224"
    )

    optimizer = AdamW(student_model.parameters(), lr=lr)

    accelerator = Accelerator()
    train_dataloader, val_dataloader, teacher_model, student_model, optimizer = (
        accelerator.prepare(
            train_dataloader, val_dataloader, teacher_model, student_model, optimizer
        )
    )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=epoch * len(train_dataloader),
    )
    metric = evaluate.load("accuracy")
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
        
    for e in range(epoch):
        # train one epoch
        student_model.train()
        for batch in train_dataloader:
            teacher_outputs = teacher_model(
                pixel_values=batch["image"],
                output_attentions=True,
                output_hidden_states=True,
            )
                        
            student_outputs = student_model(
                pixel_values=batch["image"],
                labels=batch["label"],
                output_attentions=True,
                output_hidden_states=True,
            )
            
            # Placeholder
            loss = student_outputs.loss
            
            accelerator.backward(loss)
            optimizer.step()

            current_lr = lr_scheduler.get_last_lr()[0]
            # predictions = student_outputs.logits.argmax(-1)[0]    

            # metric.add_batch(predictions=predictions, references=batch["label"])
            # result = metric.compute()

            acc1, acc5 = accuracy(student_outputs.logits, batch["label"], topk=(1, 5))

            loss_meter.update(loss.item())
            acc1_meter.update(acc1.item())
            acc5_meter.update(acc5.item())
        
            print(f"Current learning rate = {current_lr}, acc1 = {acc1_meter.avg:.3f}")
                        
            lr_scheduler.step()            
            optimizer.zero_grad()
            
        # save
        student_model.save_pretrained("save_test")        
        
        # evaluate
        for batch in val_dataloader:
            student_outputs = student_model(
                pixel_values=batch["image"],
                labels=batch["label"],
                output_attentions=True,
                output_hidden_states=True,
            )        

if __name__ == "__main__":
    main()
