---
language: sanskrit
---


# ALBERT-base-Sanskrit


Explaination Notebook Colab: [SanskritALBERT.ipynb](https://colab.research.google.com/github/parmarsuraj99/suraj-parmar/blob/master/_notebooks/2020-05-02-SanskritALBERT.ipynb)

Size of the model is **46MB**

Example of usage:

```
tokenizer = AutoTokenizer.from_pretrained("surajp/albert-base-sanskrit")
model = AutoModel.from_pretrained("surajp/albert-base-sanskrit")

enc=tokenizer.encode("ॐ सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः । सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत् । ॐ शान्तिः शान्तिः शान्तिः ॥")
print(tokenizer.decode(enc))

ps = model(torch.tensor(enc).unsqueeze(1))
print(ps[0].shape)
```
```
'''
Output:
--------
[CLS] ॐ सर्वे भवन्तु सुखिनः सर्वे सन्तु निरामयाः । सर्वे भद्राणि पश्यन्तु मा कश्चिद्दुःखभाग्भवेत् । ॐ शान्तिः शान्तिः शान्तिः ॥[SEP]
torch.Size([28, 1, 768])
```


> Created by [Suraj Parmar/@parmarsuraj99](https://twitter.com/parmarsuraj99)

> Made with <span style="color: #e25555;">&hearts;</span> in India
