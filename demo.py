from transformers import BeitImageProcessor, BeitImageProcessorFast

im_pro = BeitImageProcessor(size={"height":20, "width":20})
im_pro_fast = BeitImageProcessorFast(size={"height":20, "width": 20})

print(im_pro)
print(im_pro_fast)