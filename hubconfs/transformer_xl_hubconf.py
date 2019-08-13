from pytorch_transformers.tokenization_transfo_xl import TransfoXLTokenizer
from pytorch_transformers.modeling_transfo_xl import (
    TransfoXLModel,
    TransfoXLLMHeadModel
)

# A lot of models share the same param doc. Use a decorator
# to save typing
transformer_xl_docstring = """
    Transformer XL use a relative positioning (with sinusiodal patterns) and adaptive softmax inputs which means that:
    - you don't need to specify positioning embeddings indices
    - the tokens in the vocabulary have to be sorted to decreasing frequency.

    Params:
        pretrained_model_name_or_path: either:
            - a str with the name of a pre-trained model to load selected in the list of:
                . `transfo-xl-wt103`
            - a path or url to a pretrained model archive containing:
                . `transfo_xl_config.json` a configuration file for the model
                . `pytorch_model.bin` a PyTorch dump of a TransfoXLModel instance
            - a path or url to a pretrained model archive containing:
                . `transfo_xl_config.json` a configuration file for the model
                . `model.chkpt` a TensorFlow checkpoint
        from_tf: should we load the weights from a locally saved TensorFlow checkpoint
        cache_dir: an optional path to a folder in which the pre-trained models will be cached.
        state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of pre-trained models
        *inputs, **kwargs: additional input for the specific TransformerXL class
"""


def _append_from_pretrained_docstring(docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + docstr
        return fn
    return docstring_decorator


def transformerXLTokenizer(*args, **kwargs):
    """
    Instantiate a Transformer-XL tokenizer adapted from Vocab class in https://github.com/kimiyoung/transformer-xl

    Args:
    pretrained_model_name_or_path: Path to pretrained model archive
                                   or one of pre-trained vocab configs below.
                                       * transfo-xl-wt103

    Example:
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'transformerXLTokenizer', 'transfo-xl-wt103')
        
        text = "Who was Jim Henson ?"
        tokenized_text = tokenizer.tokenize(tokenized_text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    """
    tokenizer = TransfoXLTokenizer.from_pretrained(*args, **kwargs)
    return tokenizer


@_append_from_pretrained_docstring(transformer_xl_docstring)
def transformerXLModel(*args, **kwargs):
    """
    transformerXLModel is the basic Transformer XL model.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'transformerXLTokenizer', 'transfo-xl-wt103')

        #  Prepare tokenized input
        text_1 = "Who was Jim Henson ?"
        text_2 = "Jim Henson was a puppeteer"
        tokenized_text_1 = tokenizer.tokenize(text_1)
        tokenized_text_2 = tokenizer.tokenize(text_2)
        indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
        indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
        tokens_tensor_1 = torch.tensor([indexed_tokens_1])
        tokens_tensor_2 = torch.tensor([indexed_tokens_2])

        # Load transformerXLModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'transformerXLModel', 'transfo-xl-wt103')
        model.eval()

        # Predict hidden states features for each layer
        # We can re-use the memory cells in a subsequent call to attend a longer context
        with torch.no_grad():
                hidden_states_1, mems_1 = model(tokens_tensor_1)
                hidden_states_2, mems_2 = model(tokens_tensor_2, mems=mems_1)
    """
    model = TransfoXLModel.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(transformer_xl_docstring)
def transformerXLLMHeadModel(*args, **kwargs):
    """
    transformerXLModel is the basic Transformer XL model with the
    tied (pre-trained) language modeling head on top.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'transformerXLTokenizer', 'transfo-xl-wt103')

        #  Prepare tokenized input
        text_1 = "Who was Jim Henson ?"
        text_2 = "Jim Henson was a puppeteer"
        tokenized_text_1 = tokenizer.tokenize(text_1)
        tokenized_text_2 = tokenizer.tokenize(text_2)
        indexed_tokens_1 = tokenizer.convert_tokens_to_ids(tokenized_text_1)
        indexed_tokens_2 = tokenizer.convert_tokens_to_ids(tokenized_text_2)
        tokens_tensor_1 = torch.tensor([indexed_tokens_1])
        tokens_tensor_2 = torch.tensor([indexed_tokens_2])

        # Load transformerXLLMHeadModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'transformerXLLMHeadModel', 'transfo-xl-wt103')
        model.eval()

        # Predict hidden states features for each layer
        # We can re-use the memory cells in a subsequent call to attend a longer context
        with torch.no_grad():
                predictions_1, mems_1 = model(tokens_tensor_1)
                predictions_2, mems_2 = model(tokens_tensor_2, mems=mems_1)

        # Get the predicted last token
        predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        assert predicted_token == 'who'
    """
    model = TransfoXLLMHeadModel.from_pretrained(*args, **kwargs)
    return model
