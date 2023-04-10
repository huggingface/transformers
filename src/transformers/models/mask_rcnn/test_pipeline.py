from transformers import pipeline


pipe = pipeline(task="object-detection", model="nielsr/convnext-tiny-maskrcnn")

pipe("https://miro.medium.com/max/1000/0*w1s81z-Q72obhE_z")
