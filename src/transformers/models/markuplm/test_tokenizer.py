from transformers import LayoutLMv2TokenizerFast, MarkupLMTokenizerFast


# slow_tokenizer = MarkupLMTokenizer.from_pretrained("microsoft/markuplm-base")

# slow_encoding = slow_tokenizer(
#     ["hello", "world"],
#     xpaths=["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span"],
#     padding="max_length",
#     max_length=20,
#     return_tensors="pt",
# )

fast_tokenizer = MarkupLMTokenizerFast.from_pretrained("microsoft/markuplm-base")

fast_tokenizer_bis = LayoutLMv2TokenizerFast.from_pretrained("microsoft/layoutlmv2-base-uncased")

fast_encoding = fast_tokenizer_bis(
    ["hello", "world", "how", "are", "you"],
    # xpaths=["/html/body/div/li[1]/div/span", "/html/body/div/li[1]/div/span", "html/body", "html/body/div"],
    nodes=[[1, 2, 3, 4] for _ in range(5)],
    padding="max_length",
    max_length=2,
    return_tensors="pt",
    return_overflowing_tokens=True,
)

# for k in slow_encoding.keys():
#     assert torch.allclose(slow_encoding[k], fast_encoding[k])
