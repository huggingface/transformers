<!--Copyright 2023 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

⚠️ Note that this file is in Markdown but contain specific syntax for our doc-builder (similar to MDX) that may not be
rendered properly in your Markdown viewer.

-->

# Loading adapters using the 🤗 PEFT library

*What are adapters and why are they are useful?*. If you are not familiar with adapters and Parameter-Efficient Fine Tuning (PEFT) approaches, we advise you to have a look at the introduction of the [🤗 PEFT announcement blogpost](https://huggingface.co/blog/peft).

PEFT methods aims to fine-tune a pretrained model while keeping the number of trainable parameters as low as possible. This is achieved by freezing the pretrained model and adding a small number of trainable parameters (the adapters) on top of it. The adapters are then trained to learn the task-specific information. This approach has been shown to be very efficient in terms of memory and compute usage, while achieving competitive results compared to full fine-tuning. 

When training adapters with PEFT, you might want to save the trained adapter and share them with the community, usually the saved checkpoints are order of magnitude smaller than the full model. 

| **Example screenshot of a PEFT adapter pushed on the Hub**                                           |
|-----------------------------------------------------------------------------------------------------------------------------|
| <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/peft/PEFT-hub-screenshot.png" width=200> | 

Above is an example of what an adapter model would look like if correctly pushed on 🤗 Hub. The repo stores adapter weights of `OPTForCausalLM` model, that should have ~700MB of full model weights. As you can see from the screenshot, the adapter weights are only ~6MB, meaning the community can benefit from fine-tuned performance out of box and easily loading adapter weights.

## Setup

Get started by installing 🤗 PEFT:

```bash
pip install peft
```

If you want to try out the brand new features, you might be interesting in installing the library from source:

```bash
pip install git+https://github.com/huggingface/peft.git
```

<!--

TODO: (@younesbelkada @stevhliu)

-   From pretrained example - make sure to tell it works for auto mapping models + non-auto mapping models.
-   Link to PEFT docs for further details
-   Trainer integration - provide small snippets 
-   8-bit / 4-bit examples ?
-->
