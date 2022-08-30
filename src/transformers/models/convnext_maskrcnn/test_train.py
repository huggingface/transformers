import torch

from transformers import ConvNextMaskRCNNForObjectDetection


model = ConvNextMaskRCNNForObjectDetection.from_pretrained("nielsr/convnext-tiny-maskrcnn")

pixel_values = torch.randn([2, 3, 448, 544])

img_metas = [
    {"img_shape": (315, 419, 3), "pad_shape": (315, 419, 3)},
    {"img_shape": (425, 544, 3), "pad_shape": (425, 544, 3)},
]

labels = dict()
labels["gt_labels"] = [torch.tensor([0]), torch.tensor([0, 0, 0])]
labels["gt_bboxes"] = [torch.randn(1, 4), torch.randn(3, 4)]
labels["gt_bboxes_ignore"] = None
labels["gt_masks"] = [torch.randn([1, 315, 419]), torch.randn([3, 425, 544])]

# forward pass
with torch.no_grad():
    outputs = model(pixel_values, img_metas=img_metas, labels=labels)
