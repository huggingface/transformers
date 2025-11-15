from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from transformers.models.fast.image_processing_fast import FastImageProcessor
from transformers.models.fast.modeling_fast import FastForSceneTextRecognition


THIS_DIR = Path(__file__).resolve().parent
# absolute path to the converted checkpoint dir
model_dir = (THIS_DIR / "converted_fast_base").resolve()  # or converted_fast_small / converted_fast_base
processor = FastImageProcessor.from_pretrained(model_dir, local_files_only=True)
model = FastForSceneTextRecognition.from_pretrained(model_dir, local_files_only=True)
model.eval()


def draw_detections(img_pil, detections):
    plt.figure(figsize=(8, 8))
    plt.imshow(img_pil)
    ax = plt.gca()

    for box in detections["boxes"]:
        if len(box) == 5 and all(isinstance(x, (int, float)) for x in box):
            xc, yc, w, h, angle = box
            pts = cv2.boxPoints(((xc, yc), (w, h), angle)).astype(int).tolist()
            pts.append(pts[0])
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-r", linewidth=2)
        elif len(box) == 4 and isinstance(box[0], (list, tuple)):
            pts = list(box) + [box[0]]  # close the loop
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "-b", linewidth=2)
        elif len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            xmin, ymin, xmax, ymax = box
            w, h = xmax - xmin, ymax - ymin
            ax.add_patch(Rectangle((xmin, ymin), w, h, fill=False, linewidth=2, edgecolor="r"))
        elif len(box) == 8 and all(isinstance(x, (int, float)) for x in box):
            xs = list(box[0::2]) + [box[0]]
            ys = list(box[1::2]) + [box[1]]
            ax.plot(xs, ys, "-g", linewidth=2)
        elif len(box) > 8 and all(isinstance(x, (int, float)) for x in box):
            xs = list(box[0::2]) + [box[0]]
            ys = list(box[1::2]) + [box[1]]
            ax.plot(xs, ys, "-g", linewidth=2)

        else:
            raise ValueError(f"Unrecognized box format: {box!r}")

    ax.axis("off")
    plt.show()


def run_one(image_path, mode="boxes"):
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    post = processor.post_process_text_detection(
        outputs,
        target_sizes=[img.size[::-1]],  # (H, W)
        output_type=mode,  # "boxes" or "polygons"
    )[0]
    draw_detections(img, post)


run_one((THIS_DIR / "stop-sign1.jpg").resolve(), mode="boxes")
run_one((THIS_DIR / "stop-sign1.jpg").resolve(), mode="polygons")

run_one((THIS_DIR / "train-schedule-display.jpg").resolve(), mode="boxes")
run_one((THIS_DIR / "train-schedule-display.jpg").resolve(), mode="polygons")

run_one((THIS_DIR / "caffe-sign.jpg").resolve(), mode="boxes")
run_one((THIS_DIR / "caffe-sign.jpg").resolve(), mode="polygons")

run_one((THIS_DIR / "billboard.jpeg").resolve(), mode="boxes")
run_one((THIS_DIR / "billboard.jpeg").resolve(), mode="polygons")

run_one((THIS_DIR / "restaurant-menu.webp").resolve(), mode="boxes")
run_one((THIS_DIR / "restaurant-menu.webp").resolve(), mode="polygons")

run_one((THIS_DIR / "waterloo-sign.png").resolve(), mode="boxes")
run_one((THIS_DIR / "waterloo-sign.png").resolve(), mode="polygons")
