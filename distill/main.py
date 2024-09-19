import os, click
import numpy as np
from pathlib import Path

# from datasets import load_dataset
from torchvision import datasets
from transformers import get_scheduler, ViTImageProcessor, ViTForImageClassification
from accelerate import Accelerator
from torch.optim import AdamW
import torch
from torch.utils.data import DataLoader
from timm.utils import accuracy, AverageMeter


def get_dataloaders(dataset, batch_size):

    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def collate_fn(batch):
        images, labels = zip(*batch)
        # Some of images in ImageNet turned out to be of one channel (gray).
        images = [img.convert("RGB") for img in images]
        images = image_processor(images=images, return_tensors="pt")["pixel_values"]
        labels = torch.tensor(labels)
        return images, labels

    train_dataset = datasets.ImageFolder(os.path.join(dataset, "train"))
    val_dataset = datasets.ImageFolder(os.path.join(dataset, "val"))

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    return train_dataloader, val_dataloader


def train_one_epoch(
    epoch,
    teacher_model,
    student_model,
    train_dataloader,
    accelerator,
    optimizer,
    lr_scheduler,
):

    print_freq = max(100, int(len(train_dataloader) / 1000))
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    student_model.train()
    for idx, (images, labels) in enumerate(train_dataloader):
        teacher_outputs = teacher_model(
            pixel_values=images,
            output_attentions=True,
            output_hidden_states=True,
        )

        student_outputs = student_model(
            pixel_values=images,
            labels=labels,
            output_attentions=True,
            output_hidden_states=True,
        )

        # Placeholder
        loss = student_outputs.loss

        accelerator.backward(loss)
        optimizer.step()

        current_lr = lr_scheduler.get_last_lr()[0]
        (acc1,) = accuracy(student_outputs.logits, labels, topk=(1,))
        
        loss_meter.update(loss.item())
        acc1_meter.update(acc1.item())

        if accelerator.is_main_process and (idx % print_freq == 0):
            print(
                f"[{epoch}:{idx}] loss = {loss_meter.val:.4f} ({loss_meter.avg:.4f}), acc@1 = {acc1_meter.val:.3f}% ({acc1_meter.avg:.3f}%), learning rate = {current_lr:.5f}"
            )

        lr_scheduler.step()
        optimizer.zero_grad()


@torch.no_grad()
def evaluate(epoch, student_model, val_dataloader, accelerator):

    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    student_model.eval()
    for idx, (images, labels) in enumerate(val_dataloader):
        student_outputs = student_model(
            pixel_values=images,
            labels=labels,
            output_attentions=False,
            output_hidden_states=False,
        )

        acc1, acc5 = accuracy(student_outputs.logits, labels, topk=(1, 5))
        acc1_meter.update(acc1.item())
        acc5_meter.update(acc5.item())

        print(
            f"[{epoch}:{idx}] acc@1 = {acc1_meter.val:.3f}% ({acc1_meter.avg:.3f}%), acc@5 = {acc5_meter.val:.3f}% ({acc5_meter.avg:.3f}%)"
        )

    acc1 = accelerator.gather_for_metrics([acc1_meter.avg])
    acc5 = accelerator.gather_for_metrics([acc5_meter.avg])
    if accelerator.is_main_process:
        print(
            f"\n[Epoch {epoch} evaluation] acc@1 = {np.mean(acc1):.3f}%, acc@5 = {np.mean(acc5):.3f}%\n"
        )


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

    for e in range(epoch):
        # training
        train_one_epoch(
            e,
            teacher_model,
            student_model,
            train_dataloader,
            accelerator,
            optimizer,
            lr_scheduler,
        )

        # save both hugginface and pytorch checkpoints        
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            unwrapped_student_model = accelerator.unwrap_model(student_model)
            unwrapped_student_model.save_pretrained(save)
            torch.save(
                unwrapped_student_model.state_dict(),
                os.path.join(save, f"student-vit-base-patch16-224-epoch{e}.pth"),
            )
            print(f"\nEpoch {e} checkpoints saved.\n")

        # evaluation
        evaluate(e, student_model, val_dataloader, accelerator)


if __name__ == "__main__":
    main()
