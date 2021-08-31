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
from onnx import numpy_helper

from generation_onnx import BARTGenerator, BARTBeamSearchGenerator

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

def post_process_results(batch, output):
    trans_all_results = []
    results = []
    for id, hypos in zip(batch["id"].tolist(), output):
        results.append((id, hypos))
    batched_hypos = [hypos for _, hypos in sorted(results, key=lambda x: x[0])]
    trans_all_results.extend([task.target_dictionary.string(hypos) for hypos in batched_hypos])

    if main_args.spm_path is not None:
        trans_remove_bpe_results = [''.join(line.replace('<pad>', '').strip().split()).replace('▁', ' ').strip() for line in trans_all_results]
    elif main_args.bpe_path is not None:
        trans_remove_bpe_results = [line.replace('@@ ', '').replace('<pad>', '').strip() for line in trans_all_results]
    return trans_remove_bpe_results

data_size = len(conll14_bpe_sents)
batch_size = 32
data_lines = conll14_bpe_sents
max_length = 200
num_beams = 3
huggingface_model.config.no_repeat_ngram_size = 0
huggingface_model.config.forced_bos_token_id = None
huggingface_model.config.min_length = 0


huggingface_model.eval()

with torch.no_grad():
    start_idx = 0
    batch_lines = [line for line in data_lines[start_idx: min(start_idx + batch_size, data_size)]]
    batch_ids = [dic.encode_line(sentence, add_if_not_exist=False).long() for sentence in batch_lines]
    lengths = torch.LongTensor([t.numel() for t in batch_ids])
    batch_dataset = task.build_dataset_for_inference(batch_ids, lengths)
    batch_dataset.left_pad_source = False
    batch = batch_dataset.collater(batch_dataset)
    batch = utils.apply_to_sample(lambda t: t.to(device), batch)

    # Test export here.
    input_ids = batch['net_input']['src_tokens']
    attention_mask = batch['net_input']['src_tokens'].ne(task.source_dictionary.pad())

    summaries = huggingface_model.generate(input_ids,
                attention_mask=attention_mask,
                num_beams=1,
                max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                early_stopping=False,
                decoder_start_token_id=task.source_dictionary.eos(),
            )
    trans_remove_bpe_results = post_process_results(batch, summaries)

    # Export greedy search to ONNX
    model_path = os.path.expanduser(main_args.model_dir) + '/model.onnx'
    onnx_bart = torch.jit.script(BARTGenerator(huggingface_model))
    torch.onnx.export(onnx_bart,
        (input_ids, attention_mask, max_length+2, task.source_dictionary.eos()),
        model_path,
        opset_version=14,
        input_names=['input_ids', 'attention_mask', 'max_length', 'decoder_start_token_id'],
        output_names = ['output_ids'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 1: 'seq'},
            'output_ids': {0: 'batch', 1: 'seq_out'},
        },
        verbose=True,
        example_outputs=summaries)

    ort_sess_greedy = onnxruntime.InferenceSession(model_path)

    ort_inputs = {
        'input_ids': input_ids.cpu().numpy(),
        'attention_mask': attention_mask.cpu().numpy(),
        'max_length': np.array(max_length+2),
        'decoder_start_token_id': np.array(task.source_dictionary.eos()),
    }
    ort_out = ort_sess_greedy.run(None, ort_inputs)
    trans_remove_bpe_results_ort = post_process_results(batch, ort_out[0])

    # NOTE: below is not required when input_ids does not start with <s>
    # # Adjust inputs for beam search
    # # Remove the first id since it('<s>') was not present in the inputs during fairseq-bart-gec training
    # input_ids = input_ids[:, 1:]
    # attention_mask = attention_mask[:, 1:]

    # Compute beam search reference
    summaries = huggingface_model.generate(input_ids,
                attention_mask=attention_mask,
                num_beams=num_beams,
                max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                early_stopping=True,
                decoder_start_token_id=task.source_dictionary.eos(),
            )
    print('pt beam:', summaries)
    trans_remove_bpe_results_beam_search = post_process_results(batch, summaries)


    # Export beam search to ONNX
    model_path = os.path.expanduser(main_args.model_dir) + '/model_beam_search.onnx'
    onnx_bart = torch.jit.script(BARTBeamSearchGenerator(huggingface_model))
    torch.onnx.export(onnx_bart,
        (input_ids, attention_mask, num_beams, max_length+2, task.source_dictionary.eos()),
        model_path,
        opset_version=14,
        input_names=['input_ids', 'attention_mask', 'num_beams', 'max_length', 'decoder_start_token_id'],
        output_names = ['output_ids'],
        dynamic_axes={
            'input_ids': {0: 'batch', 1: 'seq'},
            'attention_mask': {0: 'batch', 1: 'seq'},
            'output_ids': {0: 'batch', 1: 'seq_out'},
        },
        verbose=True,
        example_outputs=summaries)

    ort_sess_beam_search = onnxruntime.InferenceSession(model_path)

    ort_inputs = {
        'input_ids': input_ids.cpu().numpy(),
        'attention_mask': attention_mask.cpu().numpy(),
        'num_beams': np.array(num_beams),
        'max_length': np.array(max_length+2),
        'decoder_start_token_id': np.array(task.source_dictionary.eos()),
    }
    ort_beam_search_out = ort_sess_beam_search.run(None, ort_inputs)
    print('ort beam:', ort_beam_search_out)
    trans_remove_bpe_results_ort_beam_search = post_process_results(batch, ort_beam_search_out[0])


    # # save inputs
    # index = 0
    # for name, data in ort_inputs.items():
    #     tensor = numpy_helper.from_array(data, name)
    #     with open('test_bart_greedy/input_{}.pb'.format(index), 'wb') as f:
    #         f.write(tensor.SerializeToString())
    #     index += 1

    # # save outputs
    # tensor = numpy_helper.from_array(ort_out[0], 'output_ids')
    # with open('test_bart_greedy/output_{}.pb'.format(0), 'wb') as f:
    #         f.write(tensor.SerializeToString())

print('pt results:', trans_remove_bpe_results)
print('ort results:', trans_remove_bpe_results_ort)
print('pt beam search results:', trans_remove_bpe_results_beam_search)
print('ort beam search results:', trans_remove_bpe_results_ort_beam_search)

print('line by line comparison:')
for p, o in zip(trans_remove_bpe_results_beam_search, trans_remove_bpe_results_ort_beam_search):
    print(p)
    print(o)