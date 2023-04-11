from transformers import MaskRCNNImageProcessor, pipeline


pipe = pipeline(
    task="object-detection", model="nielsr/convnext-tiny-maskrcnn", image_processor=MaskRCNNImageProcessor()
)

pipe("https://miro.medium.com/max/1000/0*w1s81z-Q72obhE_z")
