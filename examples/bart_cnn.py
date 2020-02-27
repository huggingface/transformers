import torch

from transformers import (
    BartForMaskedLM,
    BartTokenizer,
)


hf = BartForMaskedLM.from_pretrained("/Users/shleifer/transformers_fork/converted_cnn/", output_past=True,)
# hf.model.decoder.generation_mode = True

text = " (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian"
tok = BartTokenizer.from_pretrained("bart-large")
tokens = tok.encode(text, return_tensors="pt")
extra_len = 20
gen_tokens = hf.generate(
    tokens,
    num_return_sequences=1,
    num_beams=4,
    max_length=tokens.shape[1] + extra_len,  # repetition_penalty=10.,
    do_sample=False,
)
expected_result = (
    "<s>The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday."
)
generated = [tok.decode(g,) for g in gen_tokens]

SOURCE = "/Users/shleifer/fairseq/cnn_dm/test.source"
OUTPUT = "cnn_test_output.hypo"


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
PAD_TOKEN_ID = 1


def batch_input_ids(article_texts, device=torch_device, pad_token_id=PAD_TOKEN_ID):
    """Very gross!"""
    encoded = [tok.encode(s) for s in article_texts]
    enc_lens = [len(x) for x in encoded]

    if max(enc_lens) <= 1024:
        input_ids = tok.batch_encode_plus(article_texts, return_tensors="pt", pad_to_max_length=True)["input_ids"].to(
            device
        )
    else:
        encoded = [tok.encode_plus(s, max_length=1023)["input_ids"] for s in article_texts]
        n_pad = [max(0, 1023 - x) * [PAD_TOKEN_ID] for x in enc_lens]
        input_ids = torch.tensor([[0] + n_pad[i] + encoded[i] for i in range(len(enc_lens))],
                                 device=device, dtype=torch.long)

    # attention_mask = batch.ne(PAD_TOKEN_ID).to(device)
    return input_ids


def main(source_file=SOURCE, out_file=OUTPUT, bsz=2):
    count = 1
    with open(source_file) as source, open(out_file, "w") as fout:
        sline = source.readline().strip()
        slines = [sline]
        for sline in source:
            if count % bsz == 0:
                with torch.no_grad():
                    batch = batch_input_ids(slines)
                    hypotheses_batch = hf.generate(
                        batch,
                        num_beams=4,
                        length_penalty=2.0,
                        max_length=140 + batch.shape[1],
                        min_len=55 + batch.shape[1],
                        no_repeat_ngram_size=3,
                    )

                for hypothesis in hypotheses_batch:
                    fout.write(hypothesis + "\n")
                    fout.flush()
                slines = []

            slines.append(" " + sline.rstrip())
            count += 1
        if slines != []:
            batch = batch_input_ids(slines)
            hypotheses_batch = hf.generate(
                batch,
                num_beams=4,
                length_penalty=2.0,
                max_length=140 + batch.shape[1],
                min_len=55 + batch.shape[1],
                no_repeat_ngram_size=3,
            )
            for hypothesis in hypotheses_batch:
                fout.write(hypothesis + "\n")
                fout.flush()
