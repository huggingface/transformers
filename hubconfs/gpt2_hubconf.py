from pytorch_transformers.tokenization_gpt2 import GPT2Tokenizer
from pytorch_transformers.modeling_gpt2 import (
    GPT2Model,
    GPT2LMHeadModel,
    GPT2DoubleHeadsModel
)

# A lot of models share the same param doc. Use a decorator
# to save typing
gpt2_docstring = """
    Params:
        pretrained_model_name_or_path: either:
            - a str with the name of a pre-trained model to load selected in the list of:
                . `gpt2`, `gpt2-medium`
            - a path or url to a pretrained model archive containing:
                . `gpt2_config.json` a configuration file for the model
                . `pytorch_model.bin` a PyTorch dump of a GPT2Model instance
            - a path or url to a pretrained model archive containing:
                . `gpt2_config.json` a configuration file for the model
                . a TensorFlow checkpoint with trained weights
        from_tf: should we load the weights from a locally saved TensorFlow checkpoint
        cache_dir: an optional path to a folder in which the pre-trained models will be cached.
        state_dict: an optional state dictionary (collections.OrderedDict object) to use instead of pre-trained models
        *inputs, **kwargs: additional input for the specific GPT-2 class
"""


def _append_from_pretrained_docstring(docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + docstr
        return fn
    return docstring_decorator


def gpt2Tokenizer(*args, **kwargs):
    """
    Instantiate a GPT-2 BPE tokenizer for OpenAI GPT-2 from a pre-trained/customized vocab file.
    Peculiarities:
        - Byte-level BPE

    Args:
    pretrained_model_name_or_path: Path to pretrained model archive
                                   or one of pre-trained vocab configs below.
                                       * gpt2
    Keyword args:
    special_tokens: Special tokens in vocabulary that are not pretrained ([SEP], [CLS]...)
                    Default: None
    max_len: An artificial maximum length to truncate tokenized sequences to;
             Effective maximum length is always the minimum of this
             value (if specified) and the underlying BERT model's
             sequence length.
             Default: None

    Example:
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'gpt2Tokenizer', 'gpt2')

        text = "Who was Jim Henson ?"
        indexed_tokens = tokenizer.encode(tokenized_text)
    """
    tokenizer = GPT2Tokenizer.from_pretrained(*args, **kwargs)
    return tokenizer


@_append_from_pretrained_docstring(gpt2_docstring)
def gpt2Model(*args, **kwargs):
    """
    gpt2Model is the basic OpenAI GPT-2 Transformer model based on
    identical stacked masked self-attention blocks and pre-trained
    on large scale dataset using language modeling signal.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'gpt2Tokenizer', 'gpt2')

        #  Prepare tokenized input
        text_1 = "Who was Jim Henson ?"
        text_2 = "Jim Henson was a puppeteer"
        indexed_tokens_1 = tokenizer.encode(text_1)
        indexed_tokens_2 = tokenizer.encode(text_2)
        tokens_tensor_1 = torch.tensor([indexed_tokens_1])
        tokens_tensor_2 = torch.tensor([indexed_tokens_2])

        # Load gpt2Model
        model = torch.hub.load('huggingface/pytorch-transformers', 'gpt2Model', 'gpt2')
        model.eval()

        # Predict hidden states features for each layer
        # past can be used to reuse precomputed hidden state in a subsequent predictions
        with torch.no_grad():
                hidden_states_1, past = model(tokens_tensor_1)
                hidden_states_2, past = model(tokens_tensor_2, past=past)
    """
    model = GPT2Model.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(gpt2_docstring)
def gpt2LMHeadModel(*args, **kwargs):
    """
    gpt2LMHeadModel is the OpenAI GPT-2 Transformer model with the
    tied (pre-trained) language modeling head on top.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'gpt2Tokenizer', 'gpt2')

        #  Prepare tokenized input
        text_1 = "Who was Jim Henson ?"
        text_2 = "Jim Henson was a puppeteer"
        indexed_tokens_1 = tokenizer.encode(text_1)
        indexed_tokens_2 = tokenizer.encode(text_2)
        tokens_tensor_1 = torch.tensor([indexed_tokens_1])
        tokens_tensor_2 = torch.tensor([indexed_tokens_2])

        # Load gpt2LMHeadModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'gpt2LMHeadModel', 'gpt2')
        model.eval()

        # Predict hidden states features for each layer
        # past can be used to reuse precomputed hidden state in a subsequent predictions
        with torch.no_grad():
                predictions_1, past = model(tokens_tensor_1)
                predictions_2, past = model(tokens_tensor_2, past=past)

        # Get the predicted last token
        predicted_index = torch.argmax(predictions_2[0, -1, :]).item()
        predicted_token = tokenizer.decode([predicted_index])
        assert predicted_token == ' who'
    """
    model = GPT2LMHeadModel.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(gpt2_docstring)
def gpt2DoubleHeadsModel(*args, **kwargs):
    """
    gpt2DoubleHeadsModel is the OpenAI GPT-2 Transformer model with the
    tied (pre-trained) language modeling head and a multiple choice
    classification head (only initialized, not pre-trained).

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'gpt2Tokenizer', 'gpt2')

        #  Prepare tokenized input
        text1 = "Who was Jim Henson ? Jim Henson was a puppeteer"
        text2 = "Who was Jim Henson ? Jim Henson was a mysterious young man"
        tokenized_text1 = tokenizer.tokenize(text1)
        tokenized_text2 = tokenizer.tokenize(text2)
        indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
        indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
        tokens_tensor = torch.tensor([[indexed_tokens1, indexed_tokens2]])
        mc_token_ids = torch.LongTensor([[len(tokenized_text1)-1, len(tokenized_text2)-1]])

        # Load gpt2DoubleHeadsModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'gpt2DoubleHeadsModel', 'gpt2')
        model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
                lm_logits, multiple_choice_logits, presents = model(tokens_tensor, mc_token_ids)
    """
    model = GPT2DoubleHeadsModel.from_pretrained(*args, **kwargs)
    return model
