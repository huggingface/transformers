from pytorch_transformers.tokenization_openai import OpenAIGPTTokenizer
from pytorch_transformers.modeling_openai import (
	OpenAIGPTModel,
	OpenAIGPTLMHeadModel,
	OpenAIGPTDoubleHeadsModel
)

# Dependecies that are not specified in global hubconf.py
specific_dependencies = ['spacy', 'ftfy']

# A lot of models share the same param doc. Use a decorator
# to save typing
gpt_docstring = """
    OpenAI GPT use a single embedding matrix to store the word and special embeddings.
    Special tokens embeddings are additional tokens that are not pre-trained: [SEP], [CLS]...
    Special tokens need to be trained during the fine-tuning if you use them.
    The number of special embeddings can be controled using the `set_num_special_tokens(num_special_tokens)` function.

    The embeddings are ordered as follow in the token embeddings matrice:
        [0,                                                         ----------------------
         ...                                                        -> word embeddings
         config.vocab_size - 1,                                     ______________________
         config.vocab_size,
         ...                                                        -> special embeddings
         config.vocab_size + config.n_special - 1]                  ______________________

    where total_tokens_embeddings can be obtained as config.total_tokens_embeddings and is:
        total_tokens_embeddings = config.vocab_size + config.n_special
    You should use the associate indices to index the embeddings.

    Params:
		pretrained_model_name_or_path: either:
			- a str with the name of a pre-trained model to load selected in the list of:
				. `openai-gpt`
			- a path or url to a pretrained model archive containing:
				. `openai_gpt_config.json` a configuration file for the model
				. `pytorch_model.bin` a PyTorch dump of a OpenAIGPTModel instance
			- a path or url to a pretrained model archive containing:
				. `openai-gpt-config.json` a configuration file for the model
				. a series of NumPy files containing OpenAI TensorFlow trained weights
		from_tf: should we load the weights from a locally saved TensorFlow checkpoint
		cache_dir: an optional path to a folder in which the pre-trained models will be cached.
		state_dict: an optional state dictionary (collections.OrderedDict object)
		        	to use instead of pre-trained models
		*inputs, **kwargs: additional input for the specific OpenAI-GPT class
"""


def _append_from_pretrained_docstring(docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + docstr
        return fn
    return docstring_decorator


def openAIGPTTokenizer(*args, **kwargs):
    """
    Instantiate a BPE tokenizer for OpenAI GPT from a pre-trained/customized vocab file.
	Peculiarities:
        - lower case all inputs
        - uses SpaCy tokenizer ('en' model) and ftfy for pre-BPE tokenization if they are installed, fallback to BERT's BasicTokenizer if not.
        - argument special_tokens and function set_special_tokens:
            can be used to add additional symbols (ex: "__classify__") to a vocabulary.

    Args:
    pretrained_model_name_or_path: Path to pretrained model archive
                                   or one of pre-trained vocab configs below.
                                       * openai-gpt
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
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'openAIGPTTokenizer', 'openai-gpt')
		
		text = "Who was Jim Henson ? Jim Henson was a puppeteer"
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        [763, 509, 4265, 2298, 945, 257, 4265, 2298, 945, 509, 246, 10148, 39041, 483]
    """
    tokenizer = OpenAIGPTTokenizer.from_pretrained(*args, **kwargs)
    return tokenizer


@_append_from_pretrained_docstring(gpt_docstring)
def openAIGPTModel(*args, **kwargs):
    """
    OpenAIGPTModel is the basic OpenAI GPT Transformer model based on
	identical stacked masked self-attention blocks and pre-trained
	on large scale dataset using language modeling signal.

    Example:
        # Load the tokenizer
		import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'openAIGPTTokenizer', 'openai-gpt')

        #  Prepare tokenized input
        text = "Who was Jim Henson ? Jim Henson was a puppeteer"
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # Load openAIGPTModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'openAIGPTModel', 'openai-gpt')
        model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
                hidden_states = model(tokens_tensor)
    """
    model = OpenAIGPTModel.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(gpt_docstring)
def openAIGPTLMHeadModel(*args, **kwargs):
    """
    OpenAIGPTLMHeadModel is the OpenAI GPT Transformer model with the
	tied (pre-trained) language modeling head on top.

	Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'openAIGPTTokenizer', 'openai-gpt')

        #  Prepare tokenized input
        text = "Who was Jim Henson ? Jim Henson was a puppeteer"
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        # Load openAIGPTLMHeadModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'openAIGPTLMHeadModel', 'openai-gpt')
        model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
                predictions = model(tokens_tensor)

		# Get the predicted last token
		predicted_index = torch.argmax(predictions[0, -1, :]).item()
		predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        '.</w>'
    """
    model = OpenAIGPTLMHeadModel.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(gpt_docstring)
def openAIGPTDoubleHeadsModel(*args, **kwargs):
    """
    OpenAIGPTDoubleHeadsModel is the OpenAI GPT Transformer model with the
	tied (pre-trained) language modeling head and a multiple choice
	classification head (only initialized, not pre-trained).

	Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'openAIGPTTokenizer', 'openai-gpt')

        #  Prepare tokenized input
        text1 = "Who was Jim Henson ? Jim Henson was a puppeteer"
        text2 = "Who was Jim Henson ? Jim Henson was a mysterious young man"
        tokenized_text1 = tokenizer.tokenize(text1)
        tokenized_text2 = tokenizer.tokenize(text2)
        indexed_tokens1 = tokenizer.convert_tokens_to_ids(tokenized_text1)
        indexed_tokens2 = tokenizer.convert_tokens_to_ids(tokenized_text2)
        tokens_tensor = torch.tensor([[indexed_tokens1, indexed_tokens2]])
        mc_token_ids = torch.LongTensor([[len(tokenized_text1)-1, len(tokenized_text2)-1]])

        # Load openAIGPTDoubleHeadsModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'openAIGPTDoubleHeadsModel', 'openai-gpt')
        model.eval()

        # Predict hidden states features for each layer
        with torch.no_grad():
                lm_logits, multiple_choice_logits = model(tokens_tensor, mc_token_ids)
    """
    model = OpenAIGPTDoubleHeadsModel.from_pretrained(*args, **kwargs)
    return model
