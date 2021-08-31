from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)

import logging
import sys, os
import argparse
from fairseq.data import Dictionary
import torch
from nltk.tokenize import word_tokenize
from fairseq.tasks.translation import TranslationTask
from tqdm import tqdm
from fairseq import utils
import onnxruntime
import numpy as np

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("generate")

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", type=str, required=True)
parser.add_argument("--input_path", type=str, required=True)
parser.add_argument("--spm_path", type=str, default=None)
parser.add_argument("--bpe_path", type=str, default=None)
parser.add_argument("--vocab_path", type=str, required=True)
parser.add_argument("--output_path", type=str, required=True)
parser.add_argument("--output_path_ort", type=str, required=True)
main_args = parser.parse_args()

device = torch.device('cuda')

logger.info('Load model...')
huggingface_model = BartForConditionalGeneration.from_pretrained(os.path.expanduser(main_args.model_dir)).to(device)

logger.info('Load bpe and vocab...')
if main_args.spm_path is not None:
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(os.path.expanduser(main_args.spm_path))
    with open(os.path.expanduser(main_args.input_path), 'r') as f:
        conll14_bpe_sents = [' '.join(sp.EncodeAsPieces(l.strip())) for l in f.readlines()]
elif main_args.bpe_path is not None:
    from subword_nmt import apply_bpe
    bpe = apply_bpe.BPE(open(os.path.expanduser(main_args.bpe_path), 'r'))
    with open(os.path.expanduser(main_args.input_path), 'r') as f:
        conll14_bpe_sents = [bpe.process_line(l.strip()) for l in f.readlines()]

dic = Dictionary.load(os.path.expanduser(main_args.vocab_path))

logger.info('Gen...')
task = TranslationTask(None, dic, dic)

data_size = len(conll14_bpe_sents)
batch_size = 32
data_lines = conll14_bpe_sents
max_length = 200
trans_all_results = []
trans_all_results_ort = []
huggingface_model.config.no_repeat_ngram_size = 0
huggingface_model.config.forced_bos_token_id = None
huggingface_model.config.min_length = 0

for start_idx in tqdm(range(0, data_size, batch_size)):
    batch_lines = [line for line in data_lines[start_idx: min(start_idx + batch_size, data_size)]]
    batch_ids = [dic.encode_line(sentence, add_if_not_exist=False).long() for sentence in batch_lines]
    lengths = torch.LongTensor([t.numel() for t in batch_ids])
    batch_dataset = task.build_dataset_for_inference(batch_ids, lengths)
    batch_dataset.left_pad_source = False
    batch = batch_dataset.collater(batch_dataset)
    batch = utils.apply_to_sample(lambda t: t.to(device), batch)
    summaries = huggingface_model.generate(batch['net_input']['src_tokens'],
                attention_mask=batch['net_input']['src_tokens'].ne(task.source_dictionary.pad()),
                num_beams=3,
                max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                early_stopping=False,
                decoder_start_token_id=task.source_dictionary.eos(),
            )
    results = []
    for id, hypos in zip(batch["id"].tolist(), summaries):
        results.append((id, hypos))
    batched_hypos = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
    trans_all_results.extend([task.target_dictionary.string(hypos) for hypos in batched_hypos])

    model_path = os.path.expanduser(main_args.model_dir) + '/model.onnx'
    ort_sess_greedy = onnxruntime.InferenceSession(model_path)

    ort_inputs = {
        'input_ids': batch['net_input']['src_tokens'].cpu().numpy(),
        'attention_mask': batch['net_input']['src_tokens'].ne(task.source_dictionary.pad()).cpu().numpy(),
        'max_length': np.array(max_length+2),
        'decoder_start_token_id': np.array(task.source_dictionary.eos()),
    }
    ort_out = ort_sess_greedy.run(None, ort_inputs)

    results_ort = []
    for id, hypos in zip(batch["id"].tolist(), ort_out[0]):
        results_ort.append((id, hypos))
    batched_hypos_ort = [hypos for _, hypos in sorted(results_ort, key=lambda x: x[0])]
    trans_all_results_ort.extend([task.target_dictionary.string(hypos) for hypos in batched_hypos_ort])


def post_process_results(trans_all_results):
    if main_args.spm_path is not None:
        trans_remove_bpe_results = [''.join(line.replace('<pad>', '').strip().split()).replace('▁', ' ').strip() for line in trans_all_results]
    elif main_args.bpe_path is not None:
        trans_remove_bpe_results = [line.replace('@@ ', '').replace('<pad>', '').strip() for line in trans_all_results]
    return trans_remove_bpe_results

trans_remove_bpe_results = post_process_results(trans_all_results)
trans_remove_bpe_results_ort = post_process_results(trans_all_results_ort)

logger.info('Save...')
with open(os.path.expanduser(main_args.output_path), 'w') as f:
    for l in trans_remove_bpe_results:
        f.write(' '.join(word_tokenize(l)) + '\n')

with open(os.path.expanduser(main_args.output_path_ort), 'w') as f:
    for l in trans_remove_bpe_results_ort:
        f.write(' '.join(word_tokenize(l)) + '\n')