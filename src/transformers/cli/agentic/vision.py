# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Vision and video CLI commands.

Each function uses Auto* model classes directly (no pipeline, except
``keypoints``) and is registered as a top-level ``transformers`` CLI command
via ``app.py``.
"""

import json
from typing import Annotated

import typer

from ._common import (
    DeviceOpt,
    DtypeOpt,
    JsonOpt,
    ModelOpt,
    RevisionOpt,
    TokenOpt,
    TrustOpt,
    _load_pretrained,
    format_output,
    load_image,
    load_video,
)


def image_classify(
    image: Annotated[str, typer.Option(help="Path or URL to the image.")],
    model: ModelOpt = None,
    labels: Annotated[
        str | None, typer.Option(help="Comma-separated candidate labels for zero-shot classification.")
    ] = None,
    top_k: Annotated[int, typer.Option(help="Number of top predictions to return.")] = 5,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """Classify an image.

    Without ``--labels``, uses ``AutoModelForImageClassification`` with a
    pre-trained head (default: ``google/vit-base-patch16-224``).

    With ``--labels``, uses ``AutoModelForZeroShotImageClassification`` and
    ``AutoProcessor`` (default: ``google/siglip-base-patch16-224``).

    Example::

        transformers image-classify photo.jpg
        transformers image-classify photo.jpg --labels "cat,dog,bird"
    """
    import torch

    img = load_image(image)

    if labels is None:
        from transformers import AutoImageProcessor, AutoModelForImageClassification

        model_id = model or "google/vit-base-patch16-224"
        loaded_model, processor = _load_pretrained(
            AutoModelForImageClassification,
            AutoImageProcessor,
            model_id,
            device,
            dtype,
            trust_remote_code,
            token,
            revision,
        )
        inputs = processor(images=img, return_tensors="pt")
        if hasattr(loaded_model, "device"):
            inputs = inputs.to(loaded_model.device)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
        probs = outputs.logits.softmax(dim=-1)[0]
        top_values, top_indices = probs.topk(min(top_k, len(probs)))
        result = [
            {"label": loaded_model.config.id2label[idx.item()], "score": round(val.item(), 4)}
            for val, idx in zip(top_values, top_indices)
        ]
    else:
        from transformers import AutoModelForZeroShotImageClassification, AutoProcessor

        candidate_labels = [l.strip() for l in labels.split(",")]
        model_id = model or "google/siglip-base-patch16-224"
        loaded_model, processor = _load_pretrained(
            AutoModelForZeroShotImageClassification,
            AutoProcessor,
            model_id,
            device,
            dtype,
            trust_remote_code,
            token,
            revision,
        )
        inputs = processor(images=img, text=candidate_labels, padding=True, return_tensors="pt")
        if hasattr(loaded_model, "device"):
            inputs = inputs.to(loaded_model.device)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
        probs = outputs.logits_per_image[0].softmax(dim=-1)
        scored = [
            {"label": candidate_labels[i], "score": round(probs[i].item(), 4)} for i in range(len(candidate_labels))
        ]
        result = sorted(scored, key=lambda x: x["score"], reverse=True)

    print(format_output(result, output_json))


def detect(
    image: Annotated[str, typer.Option(help="Path or URL to the image.")],
    model: ModelOpt = None,
    text: Annotated[str | None, typer.Option(help="Text query for open-vocabulary (grounded) detection.")] = None,
    threshold: Annotated[float, typer.Option(help="Detection confidence threshold.")] = 0.5,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """Detect objects in an image.

    Without ``--text``, uses ``AutoModelForObjectDetection`` with a closed-set
    detector (default: ``PekingU/rtdetr_r18vd_coco_o365``).

    With ``--text``, uses ``AutoModelForZeroShotObjectDetection`` for
    open-vocabulary detection (default: ``IDEA-Research/grounding-dino-base``).

    Example::

        transformers detect photo.jpg
        transformers detect photo.jpg --text "cat . dog ."
    """
    import torch

    img = load_image(image)

    if text is None:
        from transformers import AutoImageProcessor, AutoModelForObjectDetection

        model_id = model or "PekingU/rtdetr_r18vd_coco_o365"
        loaded_model, processor = _load_pretrained(
            AutoModelForObjectDetection,
            AutoImageProcessor,
            model_id,
            device,
            dtype,
            trust_remote_code,
            token,
            revision,
        )
        inputs = processor(images=img, return_tensors="pt")
        if hasattr(loaded_model, "device"):
            inputs = inputs.to(loaded_model.device)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    else:
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = model or "IDEA-Research/grounding-dino-base"
        loaded_model, processor = _load_pretrained(
            AutoModelForZeroShotObjectDetection,
            AutoProcessor,
            model_id,
            device,
            dtype,
            trust_remote_code,
            token,
            revision,
        )
        inputs = processor(images=img, text=text, return_tensors="pt")
        if hasattr(loaded_model, "device"):
            inputs = inputs.to(loaded_model.device)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
        target_sizes = torch.tensor([img.size[::-1]])
        if hasattr(processor, "post_process_grounded_object_detection"):
            results = processor.post_process_grounded_object_detection(
                outputs,
                input_ids=inputs["input_ids"],
                box_threshold=threshold,
                text_threshold=threshold,
                target_sizes=target_sizes,
            )[0]
        else:
            results = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[
                0
            ]

    result = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box_coords = box.tolist()
        label_str = (
            label if isinstance(label, str) else loaded_model.config.id2label.get(label.item(), str(label.item()))
        )
        result.append(
            {
                "label": label_str,
                "score": round(score.item(), 4),
                "box": {
                    "xmin": round(box_coords[0], 1),
                    "ymin": round(box_coords[1], 1),
                    "xmax": round(box_coords[2], 1),
                    "ymax": round(box_coords[3], 1),
                },
            }
        )

    print(format_output(result, output_json))


def segment(
    image: Annotated[str, typer.Option(help="Path or URL to the image.")],
    model: ModelOpt = None,
    points: Annotated[str | None, typer.Option(help="JSON list of [x, y] points for SAM-style segmentation.")] = None,
    point_labels: Annotated[
        str | None, typer.Option(help="JSON list of point labels (1=foreground, 0=background) for SAM.")
    ] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """Segment an image.

    Without ``--points``, uses ``AutoModelForSemanticSegmentation`` for
    per-pixel class labelling (default: ``nvidia/segformer-b0-finetuned-ade-512-512``).

    With ``--points``, uses ``AutoModel`` + ``AutoProcessor`` for SAM-style
    prompted segmentation (default: ``facebook/sam-vit-base``).

    Example::

        transformers segment photo.jpg
        transformers segment photo.jpg --points '[[100, 200]]' --point-labels '[1]'
    """
    import torch

    img = load_image(image)

    if points is None:
        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        model_id = model or "nvidia/segformer-b0-finetuned-ade-512-512"
        loaded_model, processor = _load_pretrained(
            AutoModelForSemanticSegmentation,
            AutoImageProcessor,
            model_id,
            device,
            dtype,
            trust_remote_code,
            token,
            revision,
        )
        inputs = processor(images=img, return_tensors="pt")
        if hasattr(loaded_model, "device"):
            inputs = inputs.to(loaded_model.device)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
        seg_map = processor.post_process_semantic_segmentation(outputs, target_sizes=[img.size[::-1]])[0]
        total_pixels = seg_map.numel()
        unique_classes = seg_map.unique()
        result = []
        for cls_id in unique_classes:
            ratio = round((seg_map == cls_id).sum().item() / total_pixels, 4)
            label = loaded_model.config.id2label.get(cls_id.item(), str(cls_id.item()))
            result.append({"label": label, "score": ratio})
        result = sorted(result, key=lambda x: x["score"], reverse=True)
    else:
        from transformers import AutoModel, AutoProcessor

        model_id = model or "facebook/sam-vit-base"
        loaded_model, processor = _load_pretrained(
            AutoModel, AutoProcessor, model_id, device, dtype, trust_remote_code, token, revision
        )
        parsed_points = json.loads(points)
        parsed_labels = json.loads(point_labels) if point_labels else [1] * len(parsed_points)
        inputs = processor(img, input_points=[parsed_points], input_labels=[parsed_labels], return_tensors="pt")
        if hasattr(loaded_model, "device"):
            inputs = inputs.to(loaded_model.device)
        with torch.no_grad():
            outputs = loaded_model(**inputs)
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        result = {
            "num_masks": masks[0].shape[1] if len(masks) > 0 else 0,
            "iou_scores": outputs.iou_scores[0, 0].tolist(),
        }

    print(format_output(result, output_json))


def depth(
    image: Annotated[str, typer.Option(help="Path or URL to the image.")],
    model: ModelOpt = None,
    output: Annotated[str | None, typer.Option(help="Path to save the depth map as a PNG image.")] = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
):
    """Estimate a depth map from an image.

    Uses ``AutoModelForDepthEstimation`` (default:
    ``depth-anything/Depth-Anything-V2-Small-hf``).

    If ``--output`` is provided the depth map is saved as a greyscale PNG.
    Otherwise, prints the depth map dimensions.

    Example::

        transformers depth photo.jpg --output depth.png
    """
    import torch

    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    img = load_image(image)
    model_id = model or "depth-anything/Depth-Anything-V2-Small-hf"
    loaded_model, processor = _load_pretrained(
        AutoModelForDepthEstimation, AutoImageProcessor, model_id, device, dtype, trust_remote_code, token, revision
    )
    inputs = processor(images=img, return_tensors="pt")
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)
    with torch.no_grad():
        outputs = loaded_model(**inputs)

    predicted_depth = outputs.predicted_depth
    depth_map = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1)
        if predicted_depth.dim() == 2
        else predicted_depth.unsqueeze(0)
        if predicted_depth.dim() == 3
        else predicted_depth,
        size=img.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    if output is not None:
        from PIL import Image

        depth_np = depth_map.cpu().float().numpy()
        depth_min, depth_max = depth_np.min(), depth_np.max()
        if depth_max - depth_min > 0:
            depth_norm = (depth_np - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_norm = depth_np * 0.0
        depth_img = Image.fromarray(depth_norm.astype("uint8"))
        depth_img.save(output)
        print(f"Depth map saved to {output} (size: {depth_map.shape[0]}x{depth_map.shape[1]})")
    else:
        print(f"Depth map size: {depth_map.shape[0]}x{depth_map.shape[1]}")


def keypoints(
    images: Annotated[list[str], typer.Option(help="Paths to two images to match.")],
    model: ModelOpt = None,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """Match keypoints between two images.

    Uses the ``keypoint-matching`` pipeline. Requires exactly two images.

    Example::

        transformers keypoints image1.jpg image2.jpg
    """
    if len(images) != 2:
        raise SystemExit("Error: keypoints requires exactly 2 image paths.")

    from transformers import pipeline

    img1 = load_image(images[0])
    img2 = load_image(images[1])

    pipe_kwargs = {}
    if model is not None:
        pipe_kwargs["model"] = model
    if device is not None:
        pipe_kwargs["device"] = device
    if dtype != "auto":
        import torch

        pipe_kwargs["dtype"] = getattr(torch, dtype)
    if trust_remote_code:
        pipe_kwargs["trust_remote_code"] = True
    if token is not None:
        pipe_kwargs["token"] = token
    if revision is not None:
        pipe_kwargs["revision"] = revision

    pipe = pipeline("keypoint-matching", **pipe_kwargs)
    result = pipe(img1, img2)

    print(format_output(result, output_json))


def video_classify(
    video: Annotated[str, typer.Option(help="Path to the video file.")],
    model: ModelOpt = None,
    top_k: Annotated[int, typer.Option(help="Number of top predictions to return.")] = 5,
    device: DeviceOpt = None,
    dtype: DtypeOpt = "auto",
    trust_remote_code: TrustOpt = False,
    token: TokenOpt = None,
    revision: RevisionOpt = None,
    output_json: JsonOpt = False,
):
    """Classify a video.

    Uses ``AutoModelForVideoClassification`` + ``AutoImageProcessor``
    (default: ``MCG-NJU/videomae-base-finetuned-kinetics``).

    Example::

        transformers video-classify clip.mp4
    """
    import torch

    from transformers import AutoImageProcessor, AutoModelForVideoClassification

    model_id = model or "MCG-NJU/videomae-base-finetuned-kinetics"
    loaded_model, processor = _load_pretrained(
        AutoModelForVideoClassification,
        AutoImageProcessor,
        model_id,
        device,
        dtype,
        trust_remote_code,
        token,
        revision,
    )
    frames = load_video(video)
    inputs = processor(images=frames, return_tensors="pt")
    if hasattr(loaded_model, "device"):
        inputs = inputs.to(loaded_model.device)
    with torch.no_grad():
        outputs = loaded_model(**inputs)
    probs = outputs.logits.softmax(dim=-1)[0]
    top_values, top_indices = probs.topk(min(top_k, len(probs)))
    result = [
        {"label": loaded_model.config.id2label[idx.item()], "score": round(val.item(), 4)}
        for val, idx in zip(top_values, top_indices)
    ]

    print(format_output(result, output_json))
