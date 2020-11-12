#!/usr/bin/env python3
import os


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # or any {'0', '1', '2'}

import t5  # noqa: E402
from t5.data.sentencepiece_vocabulary import SentencePieceVocabulary  # noqa: E402
from transformers import T5Tokenizer  # noqa: E402
from transformers.convert_t5_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch  # noqa: E402
from transformers.modeling_t5 import T5Config, T5ForConditionalGeneration  # noqa: E402


path_to_tf_checkpoint = "/home/patrick/hugging_face/t5/mesh_tf_checkpoint"


tok = T5Tokenizer.from_pretrained("t5-small")
tok.save_pretrained(path_to_tf_checkpoint)
config = T5Config.from_pretrained("t5-small")

config.save_pretrained(path_to_tf_checkpoint)

convert_tf_checkpoint_to_pytorch(path_to_tf_checkpoint, path_to_tf_checkpoint + "/config.json", path_to_tf_checkpoint)

t5_model = t5.models.MtfModel(
    model_dir=path_to_tf_checkpoint,
    batch_size=1,
    tpu=None,
    sequence_length={"inputs": 4, "targets": 4},
)

vocab_model_path = path_to_tf_checkpoint + "/spiece.model"
vocab = SentencePieceVocabulary(vocab_model_path, extra_ids=100)

score = t5_model.score(
    inputs=["Hello there"],
    targets=["Hi I am"],
    vocabulary=vocab,
)

model = T5ForConditionalGeneration.from_pretrained(path_to_tf_checkpoint, return_dict=True)

input_ids = tok("Hello there", return_tensors="pt").input_ids
labels = tok("Hi I am", return_tensors="pt").input_ids

# input_ids and labels are ok!
loss = model(input_ids, labels=labels).loss
mtf_loss = -(labels.shape[-1] * loss.item())

assert mtf_loss - score[0][0] < 1e-4
