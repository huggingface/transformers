from transformers import  CLIPProcessor

# basically, we just change the tokenizer from CLIP to XLM-R.
class AltCLIPProcessor(CLIPProcessor):
    tokenizer_class = ("XLMRobertaTokenizer","XLMRobertaTokenizerFast")
    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
