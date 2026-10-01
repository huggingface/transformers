<!--Copyright 2024 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Personalizarea componentelor modelului

O altă modalitate de a personaliza un model este să modifici componentele acestuia în loc să scrii un model complet nou, permițându-ți să adaptezi un model la cazul tău specific de utilizare. De exemplu, poți adăuga noi layers sau optimiza mecanismul de attention al unei arhitecturi. Personalizările sunt aplicate direct unui model Transformers, așadar poți continua să folosești funcții precum [`Trainer`], [`PreTrainedModel`] și biblioteca [PEFT](https://huggingface.co/docs/peft/en/index).

Acest ghid îți va arăta cum să personalizezi mecanismul de attention al unui model pentru a-i aplica [Low-Rank Adaptation (LoRA)](https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora).

> [!TIP]
> Utilitarul [clear_import_cache](https://github.com/huggingface/transformers/blob/9985d06add07a4cc691dc54a7e34f54205c04d40/src/transformers/utils/import_utils.py#L2286) este foarte util când modifici și dezvolți iterativ codul modelului. Acesta elimină toate modulele Transformers din cache și permite Python-ului să reîncarce codul modificat fără a reporni constant mediul tău.
>
> ```py
> from transformers import AutoModel
> from transformers.utils.import_utils import clear_import_cache
>
> model = AutoModel.from_pretrained("bert-base-uncased")
> # modificări ale codului modelului
> # șterge cache-ul pentru a reîncărca codul modificat
> clear_import_cache()
> # re-importă pentru a folosi codul actualizat
> model = AutoModel.from_pretrained("bert-base-uncased")
> ```

## Clasa attention

[Segment Anything] este un model de segmentare a imaginilor care combină proiecția query-key-value (`qkv`) în mecanismele sale de attention. Pentru a reduce numărul de parametri antrenabili și overhead-ul computațional, poți aplica LoRA proiecției `qkv`. Aceasta necesită împărțirea proiecției `qkv` astfel că poți targeta separat `q` și `v` cu LoRA.

1. Creează o clasă de attention personalizată, `SamVisionAttentionSplit`, prin subclasarea clasei originale `SamVisionAttention`. În `__init__`, șterge proiecția combinată `qkv` și creează un layer liniar separat pentru `q`, `k` și `v`.

```py
import torch
import torch.nn as nn
from transformers.models.sam.modeling_sam import SamVisionAttention

class SamVisionAttentionSplit(SamVisionAttention, nn.Module):
    def __init__(self, config, window_size):
        super().__init__(config, window_size)
        # elimină proiecția combinată qkv
        del self.qkv
        # proiecții separate q, k, v
        self.q = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.k = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self.v = nn.Linear(config.hidden_size, config.hidden_size, bias=config.qkv_bias)
        self._register_load_state_dict_pre_hook(self.split_q_k_v_load_hook)
```

2. Funcția `_split_qkv_load_hook` împarte weights pre-antrenate `qkv` în weights separate `q`, `k` și `v` la încărcarea modelului pentru a asigura compatibilitatea cu orice model pre-antrenat.

```py
    def split_q_k_v_load_hook(self, state_dict, prefix, *args):
        keys_to_delete = []
        for key in list(state_dict.keys()):
            if "qkv." in key:
                # împarte q, k, v din proiecția combinată
                q, k, v = state_dict[key].chunk(3, dim=0)
                # înlocuiește cu proiecții individuale q, k, v
                state_dict[key.replace("qkv.", "q.")] = q
                state_dict[key.replace("qkv.", "k.")] = k
                state_dict[key.replace("qkv.", "v.")] = v
                # marchează vechea cheie qkv pentru ștergere
                keys_to_delete.append(key)
        
        # elimină cheile qkv vechi
        for key in keys_to_delete:
            del state_dict[key]
```

3. În forward pass, `q`, `k` și `v` sunt calculate separat în timp ce restul mecanismului de attention rămâne același.

```py
    def forward(self, hidden_states: torch.Tensor, output_attentions=False) -> torch.Tensor:
        batch_size, height, width, _ = hidden_states.shape
        qkv_shapes = (batch_size *  self.num_attention_heads,  height * width, -1)
        query = self.q(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        key = self.k(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)
        value = self.v(hidden_states).reshape((batch_size,  height * width,self.num_attention_heads, -1)).permute(0,2,1,3).reshape(qkv_shapes)

        attn_weights = (query * self.scale) @ key.transpose(-2, -1)

        attn_weights = torch.nn.functional.softmax(attn_weights, dtype=torch.float32, dim=-1).to(query.dtype)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
        attn_output = (attn_probs @ value).reshape(batch_size, self.num_attention_heads, height, width, -1)
        attn_output = attn_output.permute(0, 2, 3, 1, 4).reshape(batch_size, height, width, -1)
        attn_output = self.proj(attn_output)

        if output_attentions:
            outputs = (attn_output, attn_weights)
        else:
            outputs = (attn_output, None)
        return outputs
```

Atribuie clasa personalizată `SamVisionAttentionSplit` modulului `SamVisionAttention` al modelului original pentru a-l înlocui. Toate instanțele `SamVisionAttention` din model sunt înlocuite cu versiunea de attention împărțită.

Încarcă modelul cu [`~PreTrainedModel.from_pretrained`].

```py
from transformers import SamModel

# încarcă modelul SAM pre-antrenat
model = SamModel.from_pretrained("facebook/sam-vit-base")

# înlocuiește clasa de attention în modulul vision_encoder
for layer in model.vision_encoder.layers:
    if hasattr(layer, "attn"):
        layer.attn = SamVisionAttentionSplit(model.config.vision_config, model.config.vision_config.window_size)
```

## LoRA

Cu proiecții separate `q`, `k` și `v`, aplică LoRA la `q` și `v`.

Creează un [LoraConfig](https://huggingface.co/docs/peft/package_reference/config#peft.PeftConfig) și specifică rank-ul `r`, `lora_alpha`, `lora_dropout`, `task_type` și, cel mai important, modulele de targetat.

```py
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=16,
    lora_alpha=32,
    # aplică LoRA la q și v
    target_modules=["q", "v"],
    lora_dropout=0.1,
    task_type="FEATURE_EXTRACTION"
)
```

Pasează modelul și [LoraConfig](https://huggingface.co/docs/peft/package_reference/config#peft.PeftConfig) la [get_peft_model](https://huggingface.co/docs/peft/package_reference/peft_model#peft.get_peft_model) pentru a aplica LoRA modelului.

```py
model = get_peft_model(model, config)
```

Apelează [print_trainable_parameters](https://huggingface.co/docs/peft/package_reference/peft_model#peft.PeftMixedModel.print_trainable_parameters) pentru a vizualiza numărul de parametri pe care îi antrenezi ca rezultat față de numărul total de parametri.

```py
model.print_trainable_parameters()
"trainable params: 589,824 || all params: 94,274,096 || trainable%: 0.6256"
```
