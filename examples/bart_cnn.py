import torch

from transformers import BartForMaskedLM, BartTokenizer





SOURCE = "/Users/shleifer/fairseq/cnn_dm/test.source"
OUTPUT = "cnn_test_output.hypo"


torch_device = "cuda" if torch.cuda.is_available() else "cpu"
PAD_TOKEN_ID = 1
hf = BartForMaskedLM.from_pretrained("/Users/shleifer/transformers_fork/converted_cnn/", output_past=True,)
tok = BartTokenizer.from_pretrained("bart-large")

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
        n_pad = [max(0, 1023 - x) * [pad_token_id] for x in enc_lens]
        input_ids = torch.tensor(
            [[0] + n_pad[i] + encoded[i] for i in range(len(enc_lens))], device=device, dtype=torch.long
        )

    attention_mask = input_ids.ne(pad_token_id).to(device)
    return input_ids, attention_mask


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


if __name__ == '__main__':
    main()
