# 使い方
- transformersは、tokenizer, model, configの3クラスに分かれており、どのクラスもfrom_pretrain()から読み込みを行います。
- from_pretrainで、pathを指定して、読み込みを行ってください。

## 自分で作成したpretrainモデルを使用する場合
- 例えば、以下のようにします。
```
config = BertConfig.from_json_file('/data/language/bert/model_wiki_128/bert_config.json')
model = BertModel.from_pretrained('/data/language/bert/model_wiki_128/model.pytorch-1400000', config=config)
```

## Sentencepieceをしようする場合
- pathとしてdirを指定した場合、 `spiece.model`というファイルがあれば、そのファイルを読み込みます
- pathとしてfileを指定した場合、 拡張子が `.model`であれば、sentencepieceとして扱います。
- 例えば、以下のようにします。
```
tokenizer = BertTokenizer.from_pretrained('/data/language/bert/model_wiki_128/wiki-ja.model')
```

- sentencepiceに対応しているのはBERTのみです。(Albertはsentencepieceをtokenizerとして使用していますが、日本語のpre-train-modelがありません。)

# 日本語対応状況
- Bert-Sentencepiece
他のモデル、tokenizerは順次追加予定