from transformers import get_scheduler, ViTImageProcessor, ViTForImageClassification
from distill_vit_configuration import DistillViTConfig
import torch
import os, click
import numpy as np
from pathlib import Path
from torchvision import datasets
from torch.utils.data import DataLoader
from timm.utils import accuracy, AverageMeter

@torch.no_grad()
def evaluate(student_model, val_dataloader):

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
            f"[{idx}/{len(val_dataloader)}] acc@1 = {acc1_meter.val:.3f}% ({acc1_meter.avg:.3f}%), acc@5 = {acc5_meter.val:.3f}% ({acc5_meter.avg:.3f}%)"
        )
        break


def get_dataloaders(dataset, batch_size):
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    def collate_fn(batch):
        images, labels = zip(*batch)
        # Some of images in ImageNet turned out to be of one channel (gray).
        images = [img.convert("RGB") for img in images]
        images = image_processor(images=images, return_tensors="pt")["pixel_values"]
        labels = torch.tensor(labels)
        return images, labels

    val_dataset = datasets.ImageFolder(os.path.join(dataset, "val"))
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return val_dataloader

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
    "--checkpoint",
    type=str,
    help="Pytorch checkpoint name.",
)
@click.option(
    "--onnx",
    type=str,
    default="student-vit-base-patch16-224.onnx",
    help="ONNX file name to save.",
)
def main(dataset, batch, checkpoint, onnx):
    val_dataloader = get_dataloaders(dataset, batch)

    config = DistillViTConfig()
    model = ViTForImageClassification(config)
    model.load_state_dict(torch.load(checkpoint))
    print(model)

    evaluate(model, val_dataloader)
    
    x = torch.rand(1, 3, 224, 224).type(torch.float32)
    torch.onnx.export(
        model,
        x,
        onnx,
        export_params=True,
        opset_version=13,
        do_constant_folding=True
    )
    
if __name__ == "__main__":
    main()
    