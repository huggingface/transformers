import io
from pathlib import Path

import cv2
import gradio as gr
import matplotlib.pyplot as plt
import torch
from matplotlib.patches import Rectangle
from PIL import Image

from transformers.models.fast.image_processing_fast import FastImageProcessor
from transformers.models.fast.modeling_fast import FastForSceneTextRecognition


THIS_DIR = Path(__file__).resolve().parent
# absolute path to the converted checkpoint dir
model_dir = (THIS_DIR / "converted_fast_base").resolve()  # or converted_fast_small / converted_fast_base
processor = FastImageProcessor.from_pretrained(model_dir)
model = FastForSceneTextRecognition.from_pretrained(model_dir)
model.eval()


def draw_detections(img_pil, detections, mode="boxes"):
    plt.figure(figsize=(8, 8))
    plt.imshow(img_pil)
    ax = plt.gca()

    for box in detections["boxes"]:
        # 1) Rotated rect: (xc, yc, w, h, θ)
        if len(box) == 5 and all(isinstance(x, (int, float)) for x in box):
            xc, yc, w, h, angle = box
            pts = cv2.boxPoints(((xc, yc), (w, h), angle)).astype(int).tolist()
            pts.append(pts[0])
            xs, ys = zip(*pts)
            ax.plot(xs, ys, "-r", linewidth=2)

        # 2) FOUR corner-points polygon: [(x0,y0), (x1,y1), (x2,y2), (x3,y3)]
        elif len(box) == 4 and isinstance(box[0], (list, tuple)):
            pts = list(box) + [box[0]]  # close the loop
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ax.plot(xs, ys, "-b", linewidth=2)

        # 3) Axis-aligned bbox as 4 scalars: (xmin, ymin, xmax, ymax)
        elif len(box) == 4 and all(isinstance(x, (int, float)) for x in box):
            xmin, ymin, xmax, ymax = box
            w, h = xmax - xmin, ymax - ymin
            ax.add_patch(Rectangle((xmin, ymin), w, h, fill=False, linewidth=2, edgecolor="r"))

        # 4) 4-point polygon flattened: 8 scalars [x0,y0,…,x3,y3]
        elif len(box) == 8 and all(isinstance(x, (int, float)) for x in box):
            xs = list(box[0::2]) + [box[0]]
            ys = list(box[1::2]) + [box[1]]
            ax.plot(xs, ys, "-g", linewidth=2)

        # 5) Arbitrary polygon: >8 scalars
        elif len(box) > 8 and all(isinstance(x, (int, float)) for x in box):
            xs = list(box[0::2]) + [box[0]]
            ys = list(box[1::2]) + [box[1]]
            ax.plot(xs, ys, "-g", linewidth=2)

        else:
            raise ValueError(f"Unrecognized box format: {box!r}")

    ax.axis("off")
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    pil_img = Image.open(buf).convert("RGB")
    plt.close()
    return pil_img


def run_one(image, mode="boxes"):
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    post = processor.post_process_text_detection(outputs, target_sizes=[image.size[::-1]], output_type=mode)[0]
    return draw_detections(image, post, mode)


demo = gr.Interface(
    fn=run_one,
    inputs=[
        gr.Image(type="pil", label="Upload an image with text"),
        gr.Radio(choices=["boxes", "polygons"], value="boxes", label="Output type"),
    ],
    outputs=gr.Image(type="pil", label="Detections"),
    title="FAST Text-Detection Demo",
    description="Choose bounding-boxes or polygons and see the model overlay.",
)

if __name__ == "__main__":
    demo.launch()
