import albumentations as A
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from transformers import AutoImageProcessor, AutoModelForObjectDetection
from transformers.image_transforms import center_to_corners_format


checkpoint = "/home/ubuntu/projects/transformers/examples/pytorch/object-detection/detr-resnet-50_finetuned_cppe5_v11/checkpoint-10600"

# checkpoint = "belita/detr-resnet-50_finetuned_cppe5"
model = AutoModelForObjectDetection.from_pretrained(checkpoint)
image_processor = AutoImageProcessor.from_pretrained(checkpoint)

dataset = load_dataset("cppe-5")
eval_dataset = dataset["test"]

# DetrImageProcessor.post_process_object_detection()


eval_transform = A.Compose(
    [
        # A.Resize(400, 600),
    ],
    bbox_params=A.BboxParams(format="coco", label_fields=["category"]),
)


def format_image_annotations_as_coco(image_id, category, area, bbox):
    annotations = []
    for i in range(len(category)):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }


def augment_and_transform_batch(examples):
    images = []
    annotations = []
    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = eval_transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # apply the image processor transformations: resizing, rescaling, normalization, ...
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    return result


eval_dataset = eval_dataset.with_transform(augment_and_transform_batch)


def collate_fn(batch):
    pixel_values = [item["pixel_values"] for item in batch]
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item["labels"] for item in batch]
    batch = {}
    batch["pixel_values"] = encoding["pixel_values"]
    batch["pixel_mask"] = encoding["pixel_mask"]
    batch["labels"] = labels
    return batch


dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn)

model = model.cpu().eval()

mAP = MeanAveragePrecision(box_format="xyxy", class_metrics=True)


def convert_to_absolute_coordinates(boxes: torch.Tensor, image_size: torch.Tensor) -> torch.Tensor:
    height, width = image_size
    scale_factor = torch.stack([width, height, width, height])
    # convert shape for multiplication: (4,) -> (1, 4)
    scale_factor = scale_factor.unsqueeze(0).to(boxes.device)
    boxes = boxes * scale_factor
    return boxes


# Define our compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
# predictions and label_ids field) and has to return a dictionary string to float.
# @torch.no_grad()
# def compute_metrics(eval_pred):
#     pass

# trainer = Trainer(
#     model=model,
#     compute_metrics=compute_metrics,
#     tokenizer=image_processor,
# )

# trainer.evaluate(eval_dataset=eval_dataset)

for batch in dataloader:
    with torch.no_grad():
        # model predict boxes in YOLO format (center_x, center_y, width, height)
        # with coordinates *normalized* to [0..1] (relative coordinates)
        output = model(batch["pixel_values"].cpu())

    # For metric computation we need to collect ground truth and predicted boxes in the same format

    # 1. Collect predicted boxes, classes, scores
    # image_processor convert boxes from YOLO format to Pascal VOC format (x_min, y_min, x_max, y_max)
    # in coordinate system of an image (absolute coordinates).
    image_size = torch.stack([example["size"] for example in batch["labels"]], dim=0)
    predictions = image_processor.post_process_object_detection(output, threshold=0.1, target_sizes=image_size)

    # 2. Collect ground truth boxes in the same format for metric computation
    target = []
    for label in batch["labels"]:
        boxes = center_to_corners_format(label["boxes"])
        boxes = convert_to_absolute_coordinates(boxes, label["size"])
        labels = label["class_labels"]
        target.append({"boxes": boxes.cpu(), "labels": labels.cpu()})

    mAP.update(predictions, target)

categories = dataset["train"].features["objects"].feature["category"].names
id2label = {index: x for index, x in enumerate(categories)}

results = mAP.compute()
classes = results.pop("classes")
map_per_class = results.pop("map_per_class")
mar_100_per_class = results.pop("mar_100_per_class")
for class_id, class_map, class_mar in zip(classes, map_per_class, mar_100_per_class):
    class_name = id2label[class_id.item()]
    results[f"map_{class_name}"] = class_map
    results[f"mar_100_{class_name}"] = class_mar
results = {k: round(v.item(), 4) for k, v in results.items()}

from pprint import pprint


pprint(results)

# def get_annotations_in_coco_format(dataset):
#     """_summary_

#     Args:
#         dataset (_type_): _description_

#     Returns:
#         _type_: _description_
#     """

#     # init dataset
#     coco_dataset = {"images": [], "annotations": [], "categories": []}

#     # add label categories
#     categories = dataset.features["objects"].feature["category"]
#     for id, name in enumerate(categories.names):
#         coco_dataset["categories"].append({"id": id, "name": name, "supercategory": name})

#     # add images and annotations
#     for example in dataset:
#         coco_dataset["images"].append({
#             "file_name": f"{example['image_id']}.png",
#             "id": example["image_id"],
#             "height": example["height"],
#             "width": example["width"],
#         })
#         for i, _ in enumerate(example["objects"]["category"]):
#             coco_dataset["annotations"].append({
#                 "image_id": example["image_id"],
#                 "id": example["objects"]["id"][i],
#                 "category_id": example["objects"]["category"][i],
#                 "bbox": example["objects"]["bbox"][i],
#                 "area": example["objects"]["area"][i],
#                 "iscrowd": 0,
#             })

#     return coco_dataset


# # id2label = {index: x for index, x in enumerate(categories.names)}
# # label2id = {v: k for k, v in id2label.items()}

# class CocoDetection(torchvision.datasets.CocoDetection):
#     def __init__(self, img_folder, feature_extractor, ann_file):
#         super().__init__(img_folder, ann_file)
#         self.feature_extractor = feature_extractor

#     def __getitem__(self, idx):
#         # read in PIL image and target in COCO format
#         img, target = super(CocoDetection, self).__getitem__(idx)

#         # preprocess image and target: converting target to DETR format,
#         # resizing + normalization of both image and target)
#         image_id = self.ids[idx]
#         target = {"image_id": image_id, "annotations": target}
#         encoding = self.feature_extractor(images=img, annotations=target, return_tensors="pt")
#         pixel_values = encoding["pixel_values"].squeeze()  # remove batch dimension
#         target = encoding["labels"][0]  # remove batch dimension

#         return {"pixel_values": pixel_values, "labels": target}


# # im_processor = AutoImageProcessor.from_pretrained("MariaK/detr-resnet-50_finetuned_cppe5")

# # Transform and save dataset in COCO format
# coco_dataset = get_annotations_in_coco_format(dataset["test"])
# dataset_save_path = Path("./cppe5/test_coco.json")
# dataset_save_path.parent.mkdir(parents=True, exist_ok=True)
# dataset_save_path.write_text(json.dumps(coco_dataset, indent=True))

# test_dataset_coco_format =

# print("")
# #test_ds_coco_format = CocoDetection(path_output_cppe5, image_processor, path_anno)
