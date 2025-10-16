<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->
*This model was released on 2019-08-20 and added to Hugging Face Transformers on 2020-11-16 and contributed by [eltoto1219](https://huggingface.co/eltoto1219).*

# LXMERT

[LXMERT](https://huggingface.co/papers/1908.07490) proposes a framework to learn vision-and-language connections using a large-scale Transformer model with three encoders: object relationship, language, and cross-modality. The model is pretrained on diverse tasks including masked language modeling, masked object prediction, cross-modality matching, and image question answering, using datasets like MSCOCO, Visual-Genome, VQA 2.0, and GQA. Fine-tuned, LXMERT achieves state-of-the-art results on VQA and GQA, and improves the best result on NLVR by 22% absolute. Ablation studies and attention visualizations support the effectiveness of the model components and pretraining strategies.

<hfoptions id="usage">
<hfoption id="LxmertForQuestionAnswering">

```py
import torch
from transformers import AutoTokenizer, LxmertForQuestionAnswering

tokenizer = AutoTokenizer.from_pretrained("unc-nlp/lxmert-base-uncased")
model = LxmertForQuestionAnswering.from_pretrained("unc-nlp/lxmert-base-uncased", dtype="auto")

question, text = "How do plants create energy?", "By a process known as photosynthesis."

# Create dummy visual features (normally these would come from Faster R-CNN)
batch_size = 1
num_visual_features = 36
visual_feat_dim = 2048
visual_pos_dim = 4

visual_feats = torch.randn(batch_size, num_visual_features, visual_feat_dim)
visual_pos = torch.rand(batch_size, num_visual_features, visual_pos_dim)

inputs = tokenizer(question, text, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, visual_feats=visual_feats, visual_pos=visual_pos)

qa_scores = outputs.question_answering_score
predicted_answer_idx = qa_scores.argmax().item()

print(f"Question answering scores: {qa_scores[0]}")
print(f"Predicted answer index: {predicted_answer_idx}")
print(f"Confidence score: {qa_scores[0][predicted_answer_idx]:.4f}")
```

</hfoption>
</hfoptions>

## Usage tips

- Bounding boxes aren't necessary for visual feature embeddings. Any visual-spatial features work.
- LXMERT outputs both language and visual hidden states through the cross-modality layer, so they contain information from both modalities. Select vision or language hidden states from the first input in the tuple to access a modality that only attends to itself.
- The bidirectional cross-modality encoder attention returns attention values only when the language modality is the input and the vision modality is the context vector. The cross-modality encoder contains self-attention for each modality and cross-attention, but only cross-attention is returned. Self-attention outputs are disregarded.

## LxmertConfig

[[autodoc]] LxmertConfig

## LxmertTokenizer

[[autodoc]] LxmertTokenizer

## LxmertTokenizerFast

[[autodoc]] LxmertTokenizerFast

## Lxmert specific outputs

[[autodoc]] models.lxmert.modeling_lxmert.LxmertModelOutput

[[autodoc]] models.lxmert.modeling_lxmert.LxmertForPreTrainingOutput

[[autodoc]] models.lxmert.modeling_lxmert.LxmertForQuestionAnsweringOutput

] models.lxmert.modeling_tf_lxmert.TFLxmertForPreTrainingOutput

## LxmertModel

[[autodoc]] LxmertModel
    - forward

## LxmertForPreTraining

[[autodoc]] LxmertForPreTraining
    - forward

## LxmertForQuestionAnswering

[[autodoc]] LxmertForQuestionAnswering
    - forward

