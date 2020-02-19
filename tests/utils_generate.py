GPT2_PRETRAINED_MODEL_GENERATION_TEST_CASES = {
    #     "gpt2":
    #     "gpt2-medium":
    #     "gpt2-large":
    #     "gpt2-xl":
    "distilgpt2": {
        "seed": 0,
        "input": [[464, 3290, 318, 13779]],  # The dog is cute
        "exp_output": [
            464,
            3290,
            318,
            13779,
            996,
            339,
            460,
            3360,
            655,
            2513,
            287,
            262,
            3952,
            13,
            632,
            318,
            407,
            845,
            3621,
            284,
        ],  # The dog is cute though he can sometimes just walk in the park. It is not very nice to
    }
}

GPT2_PADDING_TOKENS = {"pad_token_id": 50256, "eos_token_id": 50256}
