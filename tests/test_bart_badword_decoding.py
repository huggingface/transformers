import torch
import pickle
from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig, set_seed
import random

set_seed(0)
def test_summarize_constraint():
    src_text = [
        """Since king james establish the rule 101, the entire kingdom has shown a substantial decrement
        of criminal cases""",
        """India postponed exams for trainee doctors which cause by the government and parlement area."""
    ]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base", add_prefix_space=True)
    # load bart tokenizer
    # with open(
    #         '/home/alvinwatner/alvin_research/topics/rc/qg/src/modified_bad_word_decoding/transformers/tests/test_samples/bart_tokenizer.obj',
    #         'rb') as file_path:
    #     tokenizer = pickle.load(file_path)

    config = BartConfig(
        encoder_layers=1,
        encoder_ffn_dim=64,
        encoder_attention_heads=4,
        decoder_layers=1,
        decoder_ffn_dim=64,
        decoder_attention_heads=4,
        d_model=128
    )

    model = BartForConditionalGeneration(config)

    input_context = "My cute dog"
    # get tokens of words that should not be generated. Note to not include the special token
    banned_words_ids = [tokenizer(ban_word)['input_ids'][1:-1] for ban_word in
                          ["visitor complain overboard", "lambda"]]
    banned_words = {'ids': banned_words_ids, 'epsilon': 1.0}
    print(f"banned_words = {banned_words}")
    print()
    # encode input context
    inputs = tokenizer(input_context, return_tensors="pt")
    # generate sequences without allowing bad_words to be generated

    outputs = model.generate(**inputs, max_length=20, banned_words= banned_words, do_sample=False)
    print(f"Raw Generated = {outputs[0]}")
    print("Generated:", tokenizer.decode(outputs[0], skip_special_tokens=True))

test_summarize_constraint()