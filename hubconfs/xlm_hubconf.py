from pytorch_transformers.tokenization_xlm import XLMTokenizer
from pytorch_transformers.modeling_xlm import (
    XLMConfig,
    XLMModel,
    XLMWithLMHeadModel,
    XLMForSequenceClassification,
    XLMForQuestionAnswering
)

# A lot of models share the same param doc. Use a decorator
# to save typing
xlm_start_docstring = """
    Model class adapted from the XLM Transformer model of
        "Cross-lingual Language Model Pretraining" by Guillaume Lample, Alexis Conneau
        Paper: https://arxiv.org/abs/1901.07291
        Original code: https://github.com/facebookresearch/XLM

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'xlmTokenizer', 'xlm-mlm-en-2048')

        #  Prepare tokenized input
        text_1 = "Who was Jim Henson ?"
        text_2 = "Jim Henson was a puppeteer"
        indexed_tokens_1 = tokenizer.encode(text_1)
        indexed_tokens_2 = tokenizer.encode(text_2)
        tokens_tensor_1 = torch.tensor([indexed_tokens_1])
        tokens_tensor_2 = torch.tensor([indexed_tokens_2])
"""

# A lot of models share the same param doc. Use a decorator
# to save typing
xlm_end_docstring = """
    Params:
        pretrained_model_name_or_path: either:
            - a str with the name of a pre-trained model to load selected in the list of:
                . `xlm-mlm-en-2048`
            - a path or url to a pretrained model archive containing:
                . `config.json` a configuration file for the model
                . `pytorch_model.bin` a PyTorch dump created using the `convert_xlm_checkpoint_to_pytorch` conversion script
        cache_dir: an optional path to a folder in which the pre-trained models will be cached.
        state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of pre-trained models
        *inputs, **kwargs: additional input for the specific XLM class
"""


def _begin_with_docstring(docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + docstr
        return fn
    return docstring_decorator

def _end_with_docstring(docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + docstr
        return fn
    return docstring_decorator


def xlmTokenizer(*args, **kwargs):
    """
    Instantiate a XLM BPE tokenizer for XLM from a pre-trained vocab file.

    Args:
    pretrained_model_name_or_path: Path to pretrained model archive
                                   or one of pre-trained vocab configs below.
                                       * xlm-mlm-en-2048
    Keyword args:
    special_tokens: Special tokens in vocabulary that are not pretrained
                    Default: None
    max_len: An artificial maximum length to truncate tokenized sequences to;
             Effective maximum length is always the minimum of this
             value (if specified) and the underlying model's
             sequence length.
             Default: None

    Example:
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'xlmTokenizer', 'xlm-mlm-en-2048')

        text = "Who was Jim Henson ?"
        indexed_tokens = tokenizer.encode(tokenized_text)
    """
    tokenizer = XLMTokenizer.from_pretrained(*args, **kwargs)
    return tokenizer


@_begin_with_docstring(xlm_start_docstring)
@_end_with_docstring(xlm_end_docstring)
def xlmModel(*args, **kwargs):
    """
        # Load xlmModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'xlmModel', 'xlm-mlm-en-2048')
        model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
                hidden_states_1, mems = model(tokens_tensor_1)
                hidden_states_2, mems = model(tokens_tensor_2, past=mems)
    """
    model = XLMModel.from_pretrained(*args, **kwargs)
    return model


@_begin_with_docstring(xlm_start_docstring)
@_end_with_docstring(xlm_end_docstring)
def xlmLMHeadModel(*args, **kwargs):
    """
        #  Prepare tokenized input
        text_1 = "Who was Jim Henson ?"
        text_2 = "Jim Henson was a puppeteer"
        indexed_tokens_1 = tokenizer.encode(text_1)
        indexed_tokens_2 = tokenizer.encode(text_2)
        tokens_tensor_1 = torch.tensor([indexed_tokens_1])
        tokens_tensor_2 = torch.tensor([indexed_tokens_2])

        # Load xlnetLMHeadModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'xlnetLMHeadModel', 'xlm-mlm-en-2048')
        model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
                predictions_1, mems = model(tokens_tensor_1)
                predictions_2, mems = model(tokens_tensor_2, mems=mems)

        # Get the predicted last token
        predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
        predicted_token = tokenizer.decode([predicted_index])
        assert predicted_token == ' who'
    """
    model = XLMWithLMHeadModel.from_pretrained(*args, **kwargs)
    return model


# @_end_with_docstring(xlnet_docstring)
# def xlnetForSequenceClassification(*args, **kwargs):
#     """
#     xlnetModel is the basic XLNet Transformer model from
#         "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
#         by Zhilin Yang, Zihang Dai1, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le

#     Example:
#         # Load the tokenizer
#         import torch
#         tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'xlnetTokenizer', 'xlm-mlm-en-2048')

#         #  Prepare tokenized input
#         text1 = "Who was Jim Henson ? Jim Henson was a puppeteer"
#         text2 = "Who was Jim Henson ? Jim Henson was a mysterious young man"
#         tokenized_text1 = tokenizer.tokenize(text1)
#         tokenized_text2 = tokenizer.tokenize(text2)
#         indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
#         indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
#         tokens_tensor = torch.tensor([[indexed_tokens1, indexed_tokens2]])
#         mc_token_ids = torch.LongTensor([[len(tokenized_text1)-1, len(tokenized_text2)-1]])

#         # Load xlnetForSequenceClassification
#         model = torch.hub.load('huggingface/pytorch-transformers', 'xlnetForSequenceClassification', 'xlm-mlm-en-2048')
#         model.eval()

#         # Predict sequence classes logits
#         with torch.no_grad():
#                 lm_logits, mems = model(tokens_tensor)
#     """
#     model = XLNetForSequenceClassification.from_pretrained(*args, **kwargs)
#     return model
