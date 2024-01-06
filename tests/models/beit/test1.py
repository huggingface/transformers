from modeling_beit import BeitModel
from modeling_tf_beit import TFBeitModel
img = Image.open('/home/madelf1337/Desktop/1.jpeg')

img_processor = BeitImageProcessor.from_pretrained("microsoft/beit-base-patch16-224")



image1 = img_processor(images = img, return_tensors='pt')
image2 = img_processor(images = img, return_tensors='tf')


pt_outputs = BeitModel(image1, output_hidden_states=True, output_attentions=True)
tf_outputs = TFBeitModel(image2, output_hidden_states=True, output_attentions=True)

print(pt_outputs)
print("--------------")
print(tf_outputs)