import unittest
import torch
import onnxruntime
import numpy as np
import os
import io
import logging
import sys

from fairseq.data import Dictionary
from fairseq.tasks.translation import TranslationTask
from fairseq import utils

from transformers import (
    AutoModel,
    AutoTokenizer,
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizer,
)

from generation_onnx import (
    BARTGenerator,
    BARTBeamSearchGenerator,
)

from tqdm import tqdm
from datasets import load_dataset


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s |  [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("generate")

class TestBART(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBART, self).__init__(*args, **kwargs)
        self.model_dir = '/home/jiz/repos/bart-onnx/test_data/demo'
        self.input_path = '/home/jiz/repos/bart-onnx/test_data/data_vocab/conll14.detok.src'
        # self.spm_path = '/home/jiz/repos/bart-onnx/test_data/data_vocab/vocab_v1.spm'
        self.spm_path = '/home/jiz/repos/bart-onnx/test_data/data_vocab/sentencepiece.bpe.model'
        self.vocab_path = '/home/jiz/repos/bart-onnx/test_data/pretrain_distill_9+3_512/dict.src.txt'
        # self.device = torch.device('cuda')
        self.device = torch.device('cpu')

        my_dataset = load_dataset('json', data_files='/home/jiz/repos/bart-onnx/test_data/data_vocab/vocab.json')

        if self.spm_path is not None:
            import sentencepiece as spm
            sp = spm.SentencePieceProcessor()
            sp.Load(os.path.expanduser(self.spm_path))
            with open(os.path.expanduser(self.input_path), 'r') as f:
                conll14_bpe_sents = [' '.join(sp.EncodeAsPieces(l.strip())) for l in f.readlines()]
        elif self.bpe_path is not None:
            from subword_nmt import apply_bpe
            bpe = apply_bpe.BPE(open(os.path.expanduser(self.bpe_path), 'r'))
            with open(os.path.expanduser(self.input_path), 'r') as f:
                conll14_bpe_sents = [bpe.process_line(l.strip()) for l in f.readlines()]

        dic = Dictionary.load(os.path.expanduser(self.vocab_path))

        self.task = TranslationTask(None, dic, dic)

        data_size = len(conll14_bpe_sents)
        batch_size = 32
        data_lines = conll14_bpe_sents
        self.max_length = 200

        self.batch_data = []

        for start_idx in tqdm(range(0, data_size, batch_size)):
            batch_lines = [line for line in data_lines[start_idx: min(start_idx + batch_size, data_size)]]
            batch_ids = [dic.encode_line(sentence, add_if_not_exist=False).long() for sentence in batch_lines]
            lengths = torch.LongTensor([t.numel() for t in batch_ids])
            batch_dataset = self.task.build_dataset_for_inference(batch_ids, lengths)
            batch_dataset.left_pad_source = False
            batch = batch_dataset.collater(batch_dataset)
            batch = utils.apply_to_sample(lambda t: t.to(self.device), batch)
            self.batch_data.append(batch)

    def test_bart_model(self):
        task = self.task
        max_length = self.max_length

        huggingface_model = BartForConditionalGeneration.from_pretrained(os.path.expanduser(self.model_dir)).to(self.device)

        huggingface_model.config.no_repeat_ngram_size = 0
        huggingface_model.config.forced_bos_token_id = None
        huggingface_model.config.min_length = 0

        huggingface_model.eval()
        ort_sess = None

        with torch.no_grad():
            for batch in self.batch_data:
                # Test export here.
                input_ids = batch['net_input']['src_tokens']
                attention_mask = batch['net_input']['src_tokens'].ne(task.source_dictionary.pad())

                outputs_forward = huggingface_model.forward(input_ids,
                            attention_mask=attention_mask,
                            num_beams=1,
                            max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                            early_stopping=False,
                            decoder_start_token_id=task.source_dictionary.eos(),
                        )

                outputs_forward_without_config_inputs = huggingface_model.forward(input_ids,
                            attention_mask,
                        )

                if not ort_sess:
                    f = io.BytesIO()
                    torch.onnx.export(huggingface_model,
                        (input_ids, attention_mask),
                        f,
                        opset_version=14,
                        input_names=['input_ids', 'attention_mask'],
                        output_names = ['logits', 'encoder_outputs'],
                        dynamic_axes={
                            'input_ids': {0: 'batch', 1: 'seq'},
                            'attention_mask': {0: 'batch', 1: 'seq'},
                            'logits': {0: 'batch', 1: 'seq_out'},
                            'encoder_outputs': {0: 'batch', 1: 'seq_out'},
                        },
                        verbose=True)
                    ort_sess = onnxruntime.InferenceSession(f.getvalue())

                ort_out = ort_sess.run(None, {
                    'input_ids': input_ids.cpu().numpy(),
                    'attention_mask': attention_mask.cpu().numpy(),
                })

                [np.testing.assert_allclose(pt_o.cpu().numpy(), ort_o, rtol=1e-3, atol=1e-3) for pt_o, ort_o in zip(outputs_forward, ort_out)]
                [np.testing.assert_allclose(pt_o.cpu().numpy(), ort_o, rtol=1e-3, atol=1e-3) for pt_o, ort_o in zip(outputs_forward_without_config_inputs, ort_out)]

    #greedy search
    def test_generator(self):
        task = self.task
        max_length = self.max_length

        huggingface_model = BartForConditionalGeneration.from_pretrained(os.path.expanduser(self.model_dir)).to(self.device)

        huggingface_model.config.no_repeat_ngram_size = 0
        huggingface_model.config.forced_bos_token_id = None
        huggingface_model.config.min_length = 0

        huggingface_model.eval()
        ort_sess = None

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.batch_data)):
                # Test export here.
                input_ids = batch['net_input']['src_tokens']
                attention_mask = batch['net_input']['src_tokens'].ne(task.source_dictionary.pad())

                summaries = huggingface_model.generate(batch['net_input']['src_tokens'],
                            attention_mask=batch['net_input']['src_tokens'].ne(task.source_dictionary.pad()),
                            num_beams=1,
                            max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                            early_stopping=False,
                            decoder_start_token_id=task.source_dictionary.eos(),
                        )

                if not ort_sess:
                    onnx_bart = torch.jit.script(BARTGenerator(huggingface_model))
                    f = io.BytesIO()
                    torch.onnx.export(onnx_bart,
                        (input_ids, attention_mask, max_length+2, task.source_dictionary.eos()),
                        f,
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
                    ort_sess = onnxruntime.InferenceSession(f.getvalue())

                ort_out = ort_sess.run(None, {
                    'input_ids': input_ids.cpu().numpy(),
                    'attention_mask': attention_mask.cpu().numpy(),
                    'max_length': np.array(max_length+2),
                    'decoder_start_token_id': np.array(task.source_dictionary.eos()),
                })

                np.testing.assert_allclose(summaries.cpu().numpy(), ort_out[0], rtol=1e-3, atol=1e-3)

    def test_beamsearch(self):
        task = self.task
        max_length = self.max_length
        num_beams = 3
        early_stopping = True

        huggingface_model = BartForConditionalGeneration.from_pretrained(os.path.expanduser(self.model_dir)).to(self.device)

        huggingface_model.config.no_repeat_ngram_size = 0
        huggingface_model.config.forced_bos_token_id = None
        huggingface_model.config.min_length = 0

        huggingface_model.eval()
        ort_sess = None

        pt_ref_model = torch.jit.script(BARTBeamSearchGenerator(huggingface_model))

        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.batch_data)):
                # Test export here.
                input_ids = batch['net_input']['src_tokens']
                attention_mask = batch['net_input']['src_tokens'].ne(task.source_dictionary.pad())

                summaries = huggingface_model.generate(batch['net_input']['src_tokens'],
                            attention_mask=batch['net_input']['src_tokens'].ne(task.source_dictionary.pad()),
                            num_beams=num_beams,
                            max_length=max_length + 2,  # +2 from original because we start at step=1 and stop before max_length
                            early_stopping=early_stopping,
                            decoder_start_token_id=task.source_dictionary.eos(),
                        )

                if not ort_sess:
                    onnx_bart = pt_ref_model
                    f = io.BytesIO()
                    torch.onnx.export(onnx_bart,
                        (input_ids, attention_mask, num_beams, max_length+2, task.source_dictionary.eos()),
                        f,
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
                    ort_sess = onnxruntime.InferenceSession(f.getvalue())

                pt_ref_out = pt_ref_model(input_ids, attention_mask, torch.tensor(num_beams), torch.tensor(max_length+2), torch.tensor(task.source_dictionary.eos()))

                ort_out = ort_sess.run(None, {
                    'input_ids': input_ids.cpu().numpy(),
                    'attention_mask': attention_mask.cpu().numpy(),
                    'num_beams': np.array(num_beams),
                    'max_length': np.array(max_length+2),
                    'decoder_start_token_id': np.array(task.source_dictionary.eos()),
                })


                np.testing.assert_allclose(summaries.cpu().numpy(), pt_ref_out.cpu().numpy(), rtol=1e-3, atol=1e-3)
                np.testing.assert_allclose(summaries.cpu().numpy(), ort_out[0], rtol=1e-3, atol=1e-3)

    def test_hug_beamsearch(self):
        local_num_beams = 3
        huggingface_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(self.device)
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        huggingface_model.eval()
        ort_sess = None

        pt_ref_model = torch.jit.script(BARTBeamSearchGenerator(huggingface_model))

        with torch.no_grad():
            ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
            input_ids = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt').to(self.device)
            
            # Generate Summary
            summary_ids = huggingface_model.generate(
                          input_ids['input_ids'],
                          attention_mask=input_ids['attention_mask'],
                          num_beams=local_num_beams,
                          max_length=5,
                          early_stopping=True,
                          decoder_start_token_id=2)
        
            if not ort_sess:
                onnx_bart = pt_ref_model
                f = io.BytesIO()
                torch.onnx.export(onnx_bart,
                    (input_ids['input_ids'], input_ids['attention_mask'], local_num_beams, 5, 2),
                    f,
                    opset_version=14,
                    input_names=['input_ids', 'attention_mask', 'num_beams', 'max_length', 'decoder_start_token_id'],
                    output_names = ['output_ids'],
                    dynamic_axes={
                        'input_ids': {0: 'batch', 1: 'seq'},
                        'output_ids': {0: 'batch', 1: 'seq_out'},
                    },
                    verbose=False,
                    example_outputs=summary_ids)
                ort_sess = onnxruntime.InferenceSession(f.getvalue())

            ort_out = ort_sess.run(None, {
                'input_ids': input_ids['input_ids'].cpu().numpy(),
                'attention_mask': input_ids['attention_mask'].cpu().numpy(),
                'num_beams': np.array(local_num_beams),
                'max_length': np.array(5),
                'decoder_start_token_id': np.array(2)
            })

            print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])
            np.testing.assert_allclose(summary_ids.cpu().numpy(), ort_out[0], rtol=1e-3, atol=1e-3)

    def test_hug_bart_model(self):
        huggingface_model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(self.device)
        tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

        huggingface_model.config.no_repeat_ngram_size = 0
        huggingface_model.config.forced_bos_token_id = None
        huggingface_model.config.min_length = 0

        huggingface_model.eval()
        ort_sess = None

        with torch.no_grad():
            ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
            input_ids = tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt').to(self.device)

            # onnx_bart = torch.jit.script(huggingface_model)
            onnx_bart = huggingface_model
            # Test export here.
            outputs_forward = huggingface_model.forward(
                          input_ids['input_ids'],
                        #   attention_mask=input_ids['attention_mask'],
                          num_beams=4,
                          max_length=5,
                          early_stopping=True)
                        #   decoder_start_token_id=2)

            outputs_forward_without_config_inputs = huggingface_model.forward(
                          input_ids['input_ids'],
                          attention_mask=input_ids['attention_mask'],
                    )

            if not ort_sess:
                f = io.BytesIO()
                torch.onnx.export(onnx_bart,
                    (input_ids['input_ids'], input_ids['attention_mask'], 4, 5, 2),
                    f,
                    opset_version=14,
                    input_names=['input_ids', 'attention_mask'],
                    output_names = ['logits', 'encoder_outputs'],
                    dynamic_axes={
                        'input_ids': {0: 'batch', 1: 'seq'},
                        'attention_mask': {0: 'batch', 1: 'seq'},
                        'logits': {0: 'batch', 1: 'seq_out'},
                        'encoder_outputs': {0: 'batch', 1: 'seq_out'},
                    },
                    do_constant_folding=False,
                    verbose=True)
                ort_sess = onnxruntime.InferenceSession(f.getvalue())

                ort_out = ort_sess.run(None, {
                    'input_ids': input_ids['input_ids'].cpu().numpy(),
                    'attention_mask': input_ids['attention_mask'].cpu().numpy(),
                })

                [np.testing.assert_allclose(pt_o.cpu().numpy(), ort_o, rtol=1e-3, atol=1e-3) for pt_o, ort_o in zip(outputs_forward, ort_out)]
                [np.testing.assert_allclose(pt_o.cpu().numpy(), ort_o, rtol=1e-3, atol=1e-3) for pt_o, ort_o in zip(outputs_forward_without_config_inputs, ort_out)]

if __name__ == '__main__':
    unittest.main()
