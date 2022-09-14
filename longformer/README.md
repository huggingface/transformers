# <p align=center>`Longformer`</p>
`Longformer` and `LongformerEncoderDecoder (LED)` are pretrained transformer models for long documents.

**\*\*\*\*\* New December 1st, 2020: LongformerEncoderDecoder \*\*\*\*\***

A `LongformerEncoderDecoder (LED)` model is now available. It supports seq2seq tasks with long input. With gradient checkpointing, fp16, and 48GB gpu, the input length can be up to 16K tokens. Check the updated paper for the model details and evaluation.

* Pretrained models:  1) [`led-base-16384`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-encdec-base-16384.tar.gz),  2) [`led-large-16384`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-encdec-large-16384.tar.gz)

* Requirements: Make sure to use the huggingface/transformers fork specified in `requirements.txt`. It adds support for gradient checkpointing and allows different maximum sequence length for the input and output. You can also run `pip install git+https://github.com/allenai/longformer.git`

* Check the script `scripts/summarization.py` for an example of how to use the model.


**\*\*\*\*\* New July 23rd, 2020: Speed degradation \*\*\*\*\***

A significant speed degradation in the hugginface/transformers was recenlty discovered and fixed (check [this PR](https://github.com/huggingface/transformers/pull/5811) for details). To avoid this problem, either use the old [release v2.11.0](https://github.com/huggingface/transformers/tree/v2.11.0) but it doesn't support gradient checkpointing, or use the master branch. This problem should be fixed with the next hugginface/transformers release.


**\*\*\*\*\* New June 29th, 2020: Easier to use Gradient checkpointing \*\*\*\*\***

Gradient checkpointing has been released with huggingface/transformers [release v3.0.0](https://github.com/huggingface/transformers/tree/v3.0.0). Gradient checkpointing reduces memory by 5x which makes it possible to process longer sequences on smaller GPUs. To use, try something like the following:

```
from transformers import LongformerModel
model = LongformerModel.from_pretrained('allenai/longformer-base-4096', gradient_checkpointing=True)
```

**\*\*\*\*\* New June 2nd, 2020: Integrating with Huggingface + Train your own long model + Gradient checkpointing \*\*\*\*\***

1. `Longformer` is now integrated in the huggingface/transformers [release v2.11.0](https://github.com/huggingface/transformers/tree/v2.11.0). Now you can do
```
from transformers import LongformerModel
model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
```
The release also includes `LongformerForQA` and other `LongformerForTaskName` with automatic setting of global attention.

2. We added a [notebook](https://colab.research.google.com/github/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb) to show how to convert an existing pretrained model into its "long" version. 

3. Gradient checkpointing has been merged into HF master ([check PR](https://github.com/huggingface/transformers/pull/4659)). Gradient checkpointing can reduce memory usage significanlty (5x for `longformer-base-4096`) allowing longer sequences on smaller gpus. 


**\*\*\*\*\* New April 27th, 2020: A PyTorch implementation of the sliding window attention  \*\*\*\*\***

We added a PyTorch implementation of the sliding window attention that doesn't require the custom CUDA kernel. It is limited in functionality but more convenient to use for finetuning on downstream tasks. 

**Advantage**: supports CPU, TPU and fp16, which aren't supported by the custom CUDA kernel

**Limitations**: uses 2x more memory (but fp16 offsets that), and doesn’t support dilation and autoregressive attention (not needed for finetuning)

therefore, it is suitable for finetuning on downstream tasks but not a good choice for language modeling. The code snippit below and the TriviaQA scripts were updated to use this new implementation.

**\*\*\*\*\* End new information \*\*\*\*\***

### How to use

1. Download pretrained model
  * [`longformer-base-4096`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-base-4096.tar.gz)
  * [`longformer-large-4096`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/longformer-large-4096.tar.gz)

2. Install environment and code

    ```bash
    conda create --name longformer python=3.7
    conda activate longformer
    conda install cudatoolkit=10.0
    pip install git+https://github.com/allenai/longformer.git
    ```

3. Run the model

    ```python
    import torch
    from longformer.longformer import Longformer, LongformerConfig
    from longformer.sliding_chunks import pad_to_window_size
    from transformers import RobertaTokenizer

    config = LongformerConfig.from_pretrained('longformer-base-4096/') 
    # choose the attention mode 'n2', 'tvm' or 'sliding_chunks'
    # 'n2': for regular n2 attantion
    # 'tvm': a custom CUDA kernel implementation of our sliding window attention
    # 'sliding_chunks': a PyTorch implementation of our sliding window attention
    config.attention_mode = 'sliding_chunks'

    model = Longformer.from_pretrained('longformer-base-4096/', config=config)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenizer.model_max_length = model.config.max_position_embeddings

    SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
 
    input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

    # TVM code doesn't work on CPU. Uncomment this if `config.attention_mode = 'tvm'`
    # model = model.cuda(); input_ids = input_ids.cuda()

    # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
    attention_mask[:, [1, 4, 21,]] =  2  # Set global attention based on the task. For example,
                                         # classification: the <s> token
                                         # QA: question tokens

    # padding seqlen to the nearest multiple of 512. Needed for the 'sliding_chunks' attention
    input_ids, attention_mask = pad_to_window_size(
            input_ids, attention_mask, config.attention_window[0], tokenizer.pad_token_id)

    output = model(input_ids, attention_mask=attention_mask)[0]
    ```

### Model pretraining

[This notebook](https://github.com/allenai/longformer/blob/master/scripts/convert_model_to_long.ipynb) demonstrates our procedure for training Longformer starting from the RoBERTa checkpoint. The same procedure can be followed to get a long-version of other existing pretrained models. 

### TriviaQA

* Training scripts: `scripts/triviaqa.py`
* Pretrained large model: [`here`](https://ai2-s2-research.s3-us-west-2.amazonaws.com/longformer/triviaqa-longformer-large.tar.gz) (replicates leaderboard results)
* Instructions: `scripts/cheatsheet.txt`


### CUDA kernel

Our custom CUDA kernel is implemented in TVM.  For now, the kernel only works on GPUs and Linux. We tested it on Ubuntu, Python 3.7, CUDA10, PyTorch >= 1.2.0. If it doesn't work for your environment, please create a new issue.

**Compiling the kernel**: We already include the compiled binaries of the CUDA kernel, so most users won't need to compile it, but if you are intersted, check `scripts/cheatsheet.txt` for instructions.


### Known issues

Please check the repo [issues](https://github.com/allenai/longformer/issues) for a list of known issues that we are planning to address soon. If your issue is not discussed, please create a new one. 


### Citing

If you use `Longformer` in your research, please cite [Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150).
```
@article{Beltagy2020Longformer,
  title={Longformer: The Long-Document Transformer},
  author={Iz Beltagy and Matthew E. Peters and Arman Cohan},
  journal={arXiv:2004.05150},
  year={2020},
}
```

`Longformer` is an open-source project developed by [the Allen Institute for Artificial Intelligence (AI2)](http://www.allenai.org).
AI2 is a non-profit institute with the mission to contribute to humanity through high-impact AI research and engineering.
