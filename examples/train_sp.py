import multiprocessing as mp
from pathlib import Path

from fire import Fire
import sentencepiece as spm


def sp_train(text: Path, model_name_prefix: str):
    spm.SentencePieceTrainer.train(
        " ".join(
            [
                f"--input={text}",
                f"--model_prefix={model_name_prefix}",
                f"--vocab_size=25000",
                f"--model_type=bpe",
                f"--max_sentence_length=50000",
                f"--unk_piece=<unk>",
                f"--control_symbols=<eot>,<cls>,<sep>,<mask>,<pad>",
                f"--character_coverage=0.9995",
                f"--shuffle_input_sentence=true",
                f"--num_threads={mp.cpu_count()}",
            ]
        )
    )

if __name__ == "__main__":
    Fire(sp_train)
