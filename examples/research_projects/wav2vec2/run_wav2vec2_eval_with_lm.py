#!/usr/bin/env python3
import itertools as it
import logging
import math
import re
import sys
import warnings
from dataclasses import dataclass, field
from typing import Optional

import datasets
import torch

import soundfile as sf
from transformers import HfArgumentParser, Wav2Vec2ForCTC, Wav2Vec2Processor


logger = logging.getLogger(__name__)


@dataclass
class EvaluationArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to evaluate.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    w2l_decoder: str = field(metadata={"help": "Use a w2l decoder."})
    test_dataset_name: Optional[str] = field(
        default="timit_asr", metadata={"help": "Specify the name of the dataset to be used."}
    )
    test_dataset_type: Optional[str] = field(
        default="clean", metadata={"help": "Specify the type of the dataset to be used."}
    )
    lexicon: Optional[str] = field(default=None, metadata={"help": "Specify the path of the lexicon file."})
    criterion: str = field(default="ctc", metadata={"help": "Define criterion type."})
    lm_weight: Optional[float] = field(
        default=0.1, metadata={"help": "Weight for lm while interpolating with neural score."}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to freeze the feature extractor layers of the model."}
    )
    verbose_logging: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to log verbose messages or not."},
    )
    unit_lm: Optional[bool] = field(default=True, metadata={"help": "Whether using unit lm or not."})
    beam: Optional[int] = field(default=200, metadata={"help": "Specify the size of the beam."})
    beam_threshold: Optional[float] = field(default=25.0, metadata={"help": "Specify the threshold for beam."})
    word_score: Optional[float] = field(
        default=1.0, metadata={"help": "Specify the score factor of a word while using lm."}
    )
    unk_weight: Optional[float] = field(default=-math.inf, metadata={"help": "Specify weight of unk token."})
    sil_weight: Optional[float] = field(default=0.0, metadata={"help": "Specify the weight of sil."})
    nbest: Optional[int] = field(default=1, metadata={"help": "Specify the number of beams to select from."})
    kenlm_model: Optional[str] = field(default=None, metadata={"help": "Specify the path of the kenlm file."})
    use_cuda: Optional[bool] = field(default=False, metadata={"help": "Whether to use cuda or not."})


def configure_logger(eval_args: EvaluationArguments):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging_level = logging.WARNING
    if eval_args.verbose_logging:
        logging_level = logging.DEBUG
    logger.setLevel(logging_level)


try:
    from flashlight.lib.sequence.criterion import CpuViterbiPath, get_data_ptr_as_bytes
    from flashlight.lib.text.decoder import (
        LM,
        CriterionType,
        KenLM,
        LexiconDecoder,
        LexiconDecoderOptions,
        LMState,
        SmearingMode,
        Trie,
    )
    from flashlight.lib.text.dictionary import create_word_dict, load_words
except Exception:
    warnings.warn(
        "flashlight python bindings are required to use this functionality. Please install from https://github.com/facebookresearch/flashlight/tree/master/bindings/python"
    )
    LM = object
    LMState = object


class W2lDecoder(object):
    def __init__(self, args, tgt_dict):
        self.tgt_dict = tgt_dict
        self.vocab_size = len(tgt_dict)
        self.nbest = args.nbest

        # criterion-specific init
        if args.criterion == "ctc":
            self.criterion_type = CriterionType.CTC
            self.blank = tgt_dict.index("<pad>") if "<pad>" in tgt_dict else tgt_dict.index("<s>")
            if "<sep>" in tgt_dict:
                self.silence = tgt_dict.index("<sep>")
            elif "|" in tgt_dict:
                self.silence = tgt_dict.index("|")
            else:
                self.silence = tgt_dict.index("</s>")
            self.asg_transitions = None
        else:
            raise RuntimeError(f"unknown criterion: {args.criterion}")

    def get_prefix(self, idxs):
        """Normalize tokens by handling CTC blank, ASG replabels, etc."""
        idxs = (g[0] for g in it.groupby(idxs))
        if self.criterion_type == CriterionType.CTC:
            idxs = filter(lambda x: x != self.blank, idxs)
        else:
            print("Only ctc criterion is supported at the moment")
            pass
        prefix_answer = ""
        for i in list(idxs):
            prefix_answer += self.tgt_dict[i]
        return prefix_answer.replace("|", " ").strip()


class W2lViterbiDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

    def decode(self, emissions):
        B, T, N = emissions.size()
        transitions = torch.FloatTensor(N, N).zero_()
        viterbi_path = torch.IntTensor(B, T)
        workspace = torch.ByteTensor(CpuViterbiPath.get_workspace_size(B, T, N))
        CpuViterbiPath.compute(
            B,
            T,
            N,
            get_data_ptr_as_bytes(emissions),
            get_data_ptr_as_bytes(transitions),
            get_data_ptr_as_bytes(viterbi_path),
            get_data_ptr_as_bytes(workspace),
        )
        return [self.get_prefix(viterbi_path[b].tolist()) for b in range(B)]


class W2lKenLMDecoder(W2lDecoder):
    def __init__(self, args, tgt_dict):
        super().__init__(args, tgt_dict)

        if args.lexicon:
            self.lexicon = load_words(args.lexicon)
            self.word_dict = create_word_dict(self.lexicon)
            self.unk_word = self.word_dict.get_index("<unk>")
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.trie = Trie(self.vocab_size, self.silence)

            start_state = self.lm.start(False)
            for i, (word, spellings) in enumerate(self.lexicon.items()):
                word_idx = self.word_dict.get_index(word)
                _, score = self.lm.score(start_state, word_idx)
                for spelling in spellings:
                    spelling_idxs = [tgt_dict.index(token) for token in spelling]
                    assert tgt_dict.index("<unk>") not in spelling_idxs, f"{spelling} {spelling_idxs}"
                    self.trie.insert(spelling_idxs, word_idx, score)
            self.trie.smear(SmearingMode.MAX)

            self.decoder_opts = LexiconDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(len(tgt_dict)),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                word_score=args.word_score,
                unk_score=args.unk_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            if self.asg_transitions is None:
                self.asg_transitions = []

            self.decoder = LexiconDecoder(
                self.decoder_opts,
                self.trie,
                self.lm,
                self.silence,
                self.blank,
                self.unk_word,
                self.asg_transitions,
                args.unit_lm,
            )
        else:
            assert args.unit_lm, "lexicon free decoding can only be done with a unit language model"
            from flashlight.lib.text.decoder import LexiconFreeDecoder, LexiconFreeDecoderOptions

            d = {w: [[w]] for w in tgt_dict}
            self.word_dict = create_word_dict(d)
            self.lm = KenLM(args.kenlm_model, self.word_dict)
            self.decoder_opts = LexiconFreeDecoderOptions(
                beam_size=args.beam,
                beam_size_token=int(len(tgt_dict)),
                beam_threshold=args.beam_threshold,
                lm_weight=args.lm_weight,
                sil_score=args.sil_weight,
                log_add=False,
                criterion_type=self.criterion_type,
            )
            self.decoder = LexiconFreeDecoder(self.decoder_opts, self.lm, self.silence, self.blank, [])

    def decode(self, emissions):
        B, T, N = emissions.size()
        hypos = []
        for b in range(B):
            emissions_ptr = emissions.data_ptr() + 4 * b * emissions.stride(0)
            print("in decode")
            print(self.decoder)
            results = self.decoder.decode(emissions_ptr, T, N)
            print("after decode")
            nbest_results = results[: self.nbest]
            hypos.append(
                [
                    {
                        "prefix": self.get_prefix(result.tokens),
                        "score": result.score,
                        "words": [self.word_dict.get_entry(x) for x in result.words if x >= 0],
                    }
                    for result in nbest_results
                ]
            )
        return hypos[0]


def main():
    parser = HfArgumentParser((EvaluationArguments))

    eval_args = parser.parse_args_into_dataclasses()
    eval_args = eval_args[0]
    configure_logger(eval_args)

    processor = Wav2Vec2Processor.from_pretrained(eval_args.model_name_or_path)
    model = Wav2Vec2ForCTC.from_pretrained(eval_args.model_name_or_path)

    def map_to_result(batch):
        if eval_args.use_cuda:
            model.to("cuda")
            input_values = processor(
                batch["speech"], sampling_rate=batch["sampling_rate"], return_tensors="pt"
            ).input_values.to("cuda")
        else:
            input_values = processor(
                batch["speech"], sampling_rate=batch["sampling_rate"], return_tensors="pt"
            ).input_values

        with torch.no_grad():
            logits = model(input_values).logits

        target_dictionary = [t for t in processor.tokenizer.get_vocab().keys()]
        if eval_args.w2l_decoder == "viterbi":
            decoder = W2lViterbiDecoder(eval_args, target_dictionary)
            batch["pred_str"] = decoder.decode(logits)[0]
        elif eval_args.w2l_decoder == "kenlm":
            decoder = W2lKenLMDecoder(eval_args, target_dictionary)
            batch["pred_str"] = decoder.decode(logits)[0]["prefix"]
        else:
            sys.exit("W2l decoder not supported.")
        return batch

    def speech_file_to_array_fn(batch):
        speech_array, sampling_rate = sf.read(batch["file"])
        batch["speech"] = speech_array
        batch["sampling_rate"] = sampling_rate
        batch["target_text"] = batch["text"]
        return batch

    selected_dataset = datasets.load_dataset(eval_args.test_dataset_name, eval_args.test_dataset_type, split="test")
    selected_dataset = selected_dataset.map(speech_file_to_array_fn, num_proc=4)
    wer_metric = datasets.load_metric("wer")
    results = selected_dataset.map(map_to_result)
    vocabulary_chars_str = "".join(t.lower() for t in processor.tokenizer.get_vocab().keys() if len(t) == 1)
    vocabulary_text_cleaner = re.compile(  # remove characters not in vocabulary
        f"[^\s{re.escape(vocabulary_chars_str)}]",  # allow space in addition to chars in vocabulary
    )
    ref = [vocabulary_text_cleaner.sub("", reference.lower()) for reference in results["target_text"]]
    pred = [vocabulary_text_cleaner.sub("", prediction.lower()) for prediction in results["pred_str"]]
    print("Test WER: {:.3f}".format(wer_metric.compute(predictions=pred, references=ref)))


if __name__ == "__main__":
    main()
