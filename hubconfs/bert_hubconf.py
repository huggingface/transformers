from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.modeling_bert import (
        BertModel,
        BertForNextSentencePrediction,
        BertForMaskedLM,
        BertForMultipleChoice,
        BertForPreTraining,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertForTokenClassification,
        )

# A lot of models share the same param doc. Use a decorator
# to save typing
bert_docstring = """
    Params:
        pretrained_model_name_or_path: either:
            - a str with the name of a pre-trained model to load
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-large-cased`
                . `bert-base-multilingual-uncased`
                . `bert-base-multilingual-cased`
                . `bert-base-chinese`
                . `bert-base-german-cased`
                . `bert-large-uncased-whole-word-masking`
                . `bert-large-cased-whole-word-masking`
            - a path or url to a pretrained model archive containing:
                . `bert_config.json` a configuration file for the model
                . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining
                  instance
            - a path or url to a pretrained model archive containing:
                . `bert_config.json` a configuration file for the model
                . `model.chkpt` a TensorFlow checkpoint
        from_tf: should we load the weights from a locally saved TensorFlow
                 checkpoint
        cache_dir: an optional path to a folder in which the pre-trained models
                   will be cached.
        state_dict: an optional state dictionary
                    (collections.OrderedDict object) to use instead of Google
                    pre-trained models
        *inputs, **kwargs: additional input for the specific Bert class
            (ex: num_labels for BertForSequenceClassification)
"""


def _append_from_pretrained_docstring(docstr):
    def docstring_decorator(fn):
        fn.__doc__ = fn.__doc__ + docstr
        return fn
    return docstring_decorator


def bertTokenizer(*args, **kwargs):
    """
    Instantiate a BertTokenizer from a pre-trained/customized vocab file
    Args:
    pretrained_model_name_or_path: Path to pretrained model archive
                                   or one of pre-trained vocab configs below.
                                       * bert-base-uncased
                                       * bert-large-uncased
                                       * bert-base-cased
                                       * bert-large-cased
                                       * bert-base-multilingual-uncased
                                       * bert-base-multilingual-cased
                                       * bert-base-chinese
    Keyword args:
    cache_dir: an optional path to a specific directory to download and cache
               the pre-trained model weights.
               Default: None
    do_lower_case: Whether to lower case the input.
                   Only has an effect when do_wordpiece_only=False
                   Default: True
    do_basic_tokenize: Whether to do basic tokenization before wordpiece.
                       Default: True
    max_len: An artificial maximum length to truncate tokenized sequences to;
             Effective maximum length is always the minimum of this
             value (if specified) and the underlying BERT model's
             sequence length.
             Default: None
    never_split: List of tokens which will never be split during tokenization.
                 Only has an effect when do_wordpiece_only=False
                 Default: ["[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]"]

    Example:
        import torch
        sentence = 'Hello, World!'
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)
        toks = tokenizer.tokenize(sentence)
        ['Hello', '##,', 'World', '##!']
        ids = tokenizer.convert_tokens_to_ids(toks)
        [8667, 28136, 1291, 28125]
    """
    tokenizer = BertTokenizer.from_pretrained(*args, **kwargs)
    return tokenizer


@_append_from_pretrained_docstring(bert_docstring)
def bertModel(*args, **kwargs):
    """
    BertModel is the basic BERT Transformer model with a layer of summed token,
    position and sequence embeddings followed by a series of identical
    self-attention blocks (12 for BERT-base, 24 for BERT-large).

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)
        #  Prepare tokenized input
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Load bertModel
        model = torch.hub.load('huggingface/pytorch-transformers', 'bertModel', 'bert-base-cased')
        model.eval()
        # Predict hidden states features for each layer
        with torch.no_grad():
                encoded_layers, _ = model(tokens_tensor, segments_tensors)
    """
    model = BertModel.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(bert_docstring)
def bertForNextSentencePrediction(*args, **kwargs):
    """
    BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence
    classification head.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)
        #  Prepare tokenized input
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Load bertForNextSentencePrediction
        model = torch.hub.load('huggingface/pytorch-transformers', 'bertForNextSentencePrediction', 'bert-base-cased')
        model.eval()
        # Predict the next sentence classification logits
        with torch.no_grad():
                next_sent_classif_logits = model(tokens_tensor, segments_tensors)
    """
    model = BertForNextSentencePrediction.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(bert_docstring)
def bertForPreTraining(*args, **kwargs):
    """
    BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads
        - the masked language modeling head, and
        - the next sentence classification head.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)
        #  Prepare tokenized input
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = tokenizer.tokenize(text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Load bertForPreTraining
        model = torch.hub.load('huggingface/pytorch-transformers', 'bertForPreTraining', 'bert-base-cased')
        masked_lm_logits_scores, seq_relationship_logits = model(tokens_tensor, segments_tensors)
    """
    model = BertForPreTraining.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(bert_docstring)
def bertForMaskedLM(*args, **kwargs):
    """
    BertForMaskedLM includes the BertModel Transformer followed by the
    (possibly) pre-trained masked language modeling head.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)
        #  Prepare tokenized input
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = tokenizer.tokenize(text)
        masked_index = 8
        tokenized_text[masked_index] = '[MASK]'
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Load bertForMaskedLM
        model = torch.hub.load('huggingface/pytorch-transformers', 'bertForMaskedLM', 'bert-base-cased')
        model.eval()
        # Predict all tokens
        with torch.no_grad():
                predictions = model(tokens_tensor, segments_tensors)
        predicted_index = torch.argmax(predictions[0, masked_index]).item()
        predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
        'henson'
    """
    model = BertForMaskedLM.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(bert_docstring)
def bertForSequenceClassification(*args, **kwargs):
    """
    BertForSequenceClassification is a fine-tuning model that includes
    BertModel and a sequence-level (sequence or pair of sequences) classifier
    on top of the BertModel. Note that the classification head is only initialized
    and has to be trained.

    The sequence-level classifier is a linear layer that takes as input the
    last hidden state of the first character in the input sequence
    (see Figures 3a and 3b in the BERT paper).

    Args:
    num_labels: the number (>=2) of classes for the classifier.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)
        #  Prepare tokenized input
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Load bertForSequenceClassification
        model = torch.hub.load('huggingface/pytorch-transformers', 'bertForSequenceClassification', 'bert-base-cased', num_labels=2)
        model.eval()
        # Predict the sequence classification logits
        with torch.no_grad():
                seq_classif_logits = model(tokens_tensor, segments_tensors)
        # Or get the sequence classification loss
        labels = torch.tensor([1])
        seq_classif_loss = model(tokens_tensor, segments_tensors, labels=labels) # set model.train() before if training this loss
    """
    model = BertForSequenceClassification.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(bert_docstring)
def bertForMultipleChoice(*args, **kwargs):
    """
    BertForMultipleChoice is a fine-tuning model that includes BertModel and a
    linear layer on top of the BertModel. Note that the multiple choice head is
    only initialized and has to be trained.

    Args:
    num_choices: the number (>=2) of classes for the classifier.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)
        #  Prepare tokenized input
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        tokens_tensor = torch.tensor([indexed_tokens, indexed_tokens]).unsqueeze(0)
        segments_tensors = torch.tensor([segments_ids, segments_ids]).unsqueeze(0)
        # Load bertForMultipleChoice
        model = torch.hub.load('huggingface/pytorch-transformers', 'bertForMultipleChoice', 'bert-base-cased', num_choices=2)
        model.eval()
        # Predict the multiple choice logits
        with torch.no_grad():
                multiple_choice_logits = model(tokens_tensor, segments_tensors)
        # Or get the multiple choice loss
        labels = torch.tensor([1])
        multiple_choice_loss = model(tokens_tensor, segments_tensors, labels=labels) # set model.train() before if training this loss
    """
    model = BertForMultipleChoice.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(bert_docstring)
def bertForQuestionAnswering(*args, **kwargs):
    """
    BertForQuestionAnswering is a fine-tuning model that includes BertModel
    with a token-level classifiers on top of the full sequence of last hidden
    states. Note that the classification head is only initialized
    and has to be trained.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)
        #  Prepare tokenized input
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Load bertForQuestionAnswering
        model = torch.hub.load('huggingface/pytorch-transformers', 'bertForQuestionAnswering', 'bert-base-cased')
        model.eval()
        # Predict the start and end positions logits
        with torch.no_grad():
                start_logits, end_logits = model(tokens_tensor, segments_tensors)
        # Or get the total loss which is the sum of the CrossEntropy loss for the start and end token positions
        start_positions, end_positions = torch.tensor([12]), torch.tensor([14])
        # set model.train() before if training this loss
        multiple_choice_loss = model(tokens_tensor, segments_tensors, start_positions=start_positions, end_positions=end_positions)
    """
    model = BertForQuestionAnswering.from_pretrained(*args, **kwargs)
    return model


@_append_from_pretrained_docstring(bert_docstring)
def bertForTokenClassification(*args, **kwargs):
    """
    BertForTokenClassification is a fine-tuning model that includes BertModel
    and a token-level classifier on top of the BertModel. Note that the classification
    head is only initialized and has to be trained.

    The token-level classifier is a linear layer that takes as input the last
    hidden state of the sequence.

    Args:
    num_labels: the number (>=2) of classes for the classifier.

    Example:
        # Load the tokenizer
        import torch
        tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'bertTokenizer', 'bert-base-cased', do_basic_tokenize=False)
        #  Prepare tokenized input
        text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
        tokenized_text = tokenizer.tokenize(text)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
        # Load bertForTokenClassification
        model = torch.hub.load('huggingface/pytorch-transformers', 'bertForTokenClassification', 'bert-base-cased', num_labels=2)
        model.eval()
        # Predict the token classification logits
        with torch.no_grad():
                classif_logits = model(tokens_tensor, segments_tensors)
        # Or get the token classification loss
        labels = torch.tensor([[0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 0]])
        classif_loss = model(tokens_tensor, segments_tensors, labels=labels) # set model.train() before if training this loss
    """
    model = BertForTokenClassification.from_pretrained(*args, **kwargs)
    return model
