from pytorch_transformers.tokenization_xlnet import XLNetTokenizer
from pytorch_transformers.modeling_xlnet import (
    XLNetConfig,
    XLNetModel,
    XLNetLMHeadModel,
    # XLNetForSequenceClassification
)

# A lot of models share the same param doc. Use a decorator
# to save typing
xlnet_docstring = """
    Params:
        pretrained_model_name_or_path: either:
            - a str with the name of a pre-trained model to load selected in the list of:
                . `xlnet-large-cased`
            - a path or url to a pretrained model archive containing:
                . `config.json` a configuration file for the model
                . `pytorch_model.bin` a PyTorch dump of a XLNetForPreTraining instance
            - a path or url to a pretrained model archive containing:
                . `xlnet_config.json` a configuration file for the model
                . `model.chkpt` a TensorFlow checkpoint
        from_tf: should we load the weights from a locally saved TensorFlow checkpoint
        cache_dir: an optional path to a folder in which the pre-trained models will be cached.
        state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of pre-trained models
        *inputs, **kwargs: additional input for the specific XLNet class
"""


def _append_from_pretrained_docstring(docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + docstr
        return fn
    return docstring_decorator


def xlnetTokenizer(*args, **kwargs):
    """
    Instantiate a XLNet sentencepiece tokenizer for XLNet from a pre-trained vocab file.
    Peculiarities:
        - require Google sentencepiece (https://github.com/google/sentencepiece)

    Args:
    pretrained_model_name_or_path: Path to pretrained model archive
                                   or one of pre-trained vocab configs below.
                                       * xlnet-large-cased
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
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'xlnetTokenizer', 'xlnet-large-cased')

        text = "Who was Jim Henson ?"
        indexed_tokens = tokenizer.encode(tokenized_text)
    """
    tokenizer = XLNetTokenizer.from_pretrained(*args, **kwargs)
    return tokenizer


@_append_from_pretrained_docstring(xlnet_docstring)
def xlnetModel(*args, **kwargs):
    """
    xlnetModel is the basic XLNet Transformer model from
        "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
        by Zhilin Yang, Zihang Dai1, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'xlnetTokenizer', 'xlnet-large-cased')

        #  Prepare tokenized input
        text_1 = "Who was Jim Henson ?"
        text_2 = "Jim Henson was a puppeteer"
        indexed_tokens_1 = tokenizer.encode(text_1)
        indexed_tokens_2 = tokenizer.encode(text_2)
        tokens_tensor_1 = torch.tensor([indexed_tokens_1])
        tokens_tensor_2 = torch.tensor([indexed_tokens_2])

        # Load xlnetModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'xlnetModel', 'xlnet-large-cased')
        model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
                hidden_states_1, mems = model(tokens_tensor_1)
                hidden_states_2, mems = model(tokens_tensor_2, past=mems)
    """
    model = XLNetModel.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(xlnet_docstring)
def xlnetLMHeadModel(*args, **kwargs):
    """
    xlnetModel is the basic XLNet Transformer model from
        "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
        by Zhilin Yang, Zihang Dai1, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le
    with a tied (pre-trained) language modeling head on top.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'xlnetTokenizer', 'xlnet-large-cased')

        #  Prepare tokenized input
        text_1 = "Who was Jim Henson ?"
        text_2 = "Jim Henson was a puppeteer"
        indexed_tokens_1 = tokenizer.encode(text_1)
        indexed_tokens_2 = tokenizer.encode(text_2)
        tokens_tensor_1 = torch.tensor([indexed_tokens_1])
        tokens_tensor_2 = torch.tensor([indexed_tokens_2])

        # Load xlnetLMHeadModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'xlnetLMHeadModel', 'xlnet-large-cased')
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
    model = XLNetLMHeadModel.from_pretrained(*args, **kwargs)
    return model


# @_append_from_pretrained_docstring(xlnet_docstring)
# def xlnetForSequenceClassification(*args, **kwargs):
#     """
#     xlnetModel is the basic XLNet Transformer model from
#         "XLNet: Generalized Autoregressive Pretraining for Language Understanding"
#         by Zhilin Yang, Zihang Dai1, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le

#     Example:
#         # Load the tokenizer
#         import torch
#         tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'xlnetTokenizer', 'xlnet-large-cased')

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
#         model = torch.hub.load('huggingface/pytorch-transformers', 'xlnetForSequenceClassification', 'xlnet-large-cased')
#         model.eval()

#         # Predict sequence classes logits
#         with torch.no_grad():
#                 lm_logits, mems = model(tokens_tensor)
#     """
#     model = XLNetForSequenceClassification.from_pretrained(*args, **kwargs)
#     return model
