---
language: zh 
widget:
- text: "[CLS] 万 叠 春 山 积 雨 晴 ，"
- text: "[CLS] 青 山 削 芙 蓉 ，"


---

# Chinese Poem GPT2 Model

## Model description

The model is used to generate Chinese ancient poems. You can download the model  either from the [GPT2-Chinese Github page](https://github.com/Morizeyao/GPT2-Chinese), or via HuggingFace from the link [gpt2-chinese-poem][poem].

Since the parameter skip_special_tokens is used in the pipelines.py, special tokens such as [SEP], [UNK] will be deleted, and the output results may not be neat.

## How to use

You can use the model directly with a pipeline for text generation:

When the parameter skip_special_tokens is True:

```python
>>> from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
>>> from transformers import TextGenerationPipeline, 
>>> tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-poem")
>>> model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-poem")
>>> text_generator = TextGenerationPipeline(model, tokenizer)   
>>> text_generator("[CLS]梅 山 如 积 翠 ，", max_length=50, do_sample=True)
	[{'generated_text': '[CLS]梅 山 如 积 翠 ， 的 手 堪 捧 。 遥 遥 仙 人 尉 ， 盘 盘 故 时 陇 。 丹 泉 清 可 鉴 ， 石 乳 甘 于 。 行 将 解 尘 缨 ， 于 焉 蹈 高 踵 。 我'}]
```

When the parameter skip_special_tokens is False:

```python
>>> from transformers import BertTokenizer, GPT2LMHeadModel, TextGenerationPipeline
>>> from transformers import TextGenerationPipeline, 
>>> tokenizer = BertTokenizer.from_pretrained("uer/gpt2-chinese-poem")
>>> model = GPT2LMHeadModel.from_pretrained("uer/gpt2-chinese-poem")
>>> text_generator = TextGenerationPipeline(model, tokenizer)   
>>> text_generator("[CLS]梅 山 如 积 翠 ，", max_length=50, do_sample=True)
	[{'generated_text': '[CLS]梅 山 如 积 翠 ， 的 [UNK] 手 堪 捧 。 遥 遥 仙 人 尉 ， 盘 盘 故 时 陇 。 丹 泉 清 可 鉴 ， 石 乳 甘 可 捧 。 银 汉 迟 不 来 ， 槎 头 欲 谁 揽 。 何'}]
```

## Training data

Contains 800,000 Chinese ancient poems collected by [chinese-poetry](https://github.com/chinese-poetry/chinese-poetry) and [Poetry](https://github.com/Werneror/Poetry) projects.

## Training procedure

The model is pre-trained by [UER-py](https://github.com/dbiir/UER-py/) on [Tencent Cloud TI-ONE](https://cloud.tencent.com/product/tione/). We pre-train 200,000 steps with a sequence length of 128.

```
python3 preprocess.py --corpus_path corpora/poem.txt \
		      --vocab_path models/google_zh_vocab.txt \  
		      --dataset_path poem.pt --processes_num 16 \
		      --seq_length 128 --target lm 
```

```
python3 pretrain.py --dataset_path poem.pt \
		    --vocab_path models/google_zh_vocab.txt \
		    --output_model_path models/poem_gpt_base_model.bin \  
		    --config_path models/bert_base_config.json --learning_rate 5e-4 \
		    --tie_weight --world_size 8 --gpu_ranks 0 1 2 3 4 5 6 7 \
		    --batch_size 64 --report_steps 1000 \
		    --save_checkpoint_steps 50000 --total_steps 200000 \
		    --embedding gpt --encoder gpt2 --target lm

```

### BibTeX entry and citation info

```
@article{zhao2019uer,
  title={UER: An Open-Source Toolkit for Pre-training Models},
  author={Zhao, Zhe and Chen, Hui and Zhang, Jinbin and Zhao, Xin and Liu, Tao and Lu, Wei and Chen, Xi and Deng, Haotang and Ju, Qi and Du, Xiaoyong},
  journal={EMNLP-IJCNLP 2019},
  pages={241},
  year={2019}
}
```

[poem]: https://huggingface.co/uer/gpt2-chinese-poem
