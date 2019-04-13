from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import (BertForNextSentencePrediction,
                                              BertForMaskedLM,
                                              BertForMultipleChoice,
                                              BertForPreTraining,
                                              BertForQuestionAnswering,
                                              BertForSequenceClassification,
                                              )

dependencies = ['torch', 'tqdm', 'boto3', 'requests', 'regex']


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
        >>> sentence = 'Hello, World!'
        >>> tokenizer = torch.hub.load('ailzhang/pytorch-pretrained-BERT:hubconf', 'BertTokenizer', 'bert-base-cased', do_basic_tokenize=False, force_reload=False)
        >>> toks = tokenizer.tokenize(sentence)
        ['Hello', '##,', 'World', '##!']
        >>> ids = tokenizer.convert_tokens_to_ids(toks)
        [8667, 28136, 1291, 28125]
    """
    tokenizer = BertTokenizer.from_pretrained(*args, **kwargs)
    return tokenizer


def bertForNextSentencePrediction(*args, **kwargs):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.
    Params:
        pretrained_model_name_or_path: either:
            - a str with the name of a pre-trained model to load selected in the list of:
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-large-cased`
                . `bert-base-multilingual-uncased`
                . `bert-base-multilingual-cased`
                . `bert-base-chinese`
            - a path or url to a pretrained model archive containing:
                . `bert_config.json` a configuration file for the model
                . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            - a path or url to a pretrained model archive containing:
                . `bert_config.json` a configuration file for the model
                . `model.chkpt` a TensorFlow checkpoint
        from_tf: should we load the weights from a locally saved TensorFlow checkpoint
        cache_dir: an optional path to a folder in which the pre-trained models will be cached.
        state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
        *inputs, **kwargs: additional input for the specific Bert class
            (ex: num_labels for BertForSequenceClassification)
    """
    model = BertForNextSentencePrediction.from_pretrained(*args, **kwargs)
    return model


def bertForPreTraining(*args, **kwargs):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.
    Params:
        pretrained_model_name_or_path: either:
            - a str with the name of a pre-trained model to load selected in the list of:
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-large-cased`
                . `bert-base-multilingual-uncased`
                . `bert-base-multilingual-cased`
                . `bert-base-chinese`
            - a path or url to a pretrained model archive containing:
                . `bert_config.json` a configuration file for the model
                . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            - a path or url to a pretrained model archive containing:
                . `bert_config.json` a configuration file for the model
                . `model.chkpt` a TensorFlow checkpoint
        from_tf: should we load the weights from a locally saved TensorFlow checkpoint
        cache_dir: an optional path to a folder in which the pre-trained models will be cached.
        state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
        *inputs, **kwargs: additional input for the specific Bert class
            (ex: num_labels for BertForSequenceClassification)

    """
    model = BertForPreTraining.from_pretrained(*args, **kwargs)
    return model


def bertForMaskedLM(*args, **kwargs):
    """
    BertForMaskedLM includes the BertModel Transformer followed by the (possibly)
    pre-trained masked language modeling head.
    Params:
        pretrained_model_name_or_path: either:
            - a str with the name of a pre-trained model to load selected in the list of:
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-large-cased`
                . `bert-base-multilingual-uncased`
                . `bert-base-multilingual-cased`
                . `bert-base-chinese`
            - a path or url to a pretrained model archive containing:
                . `bert_config.json` a configuration file for the model
                . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            - a path or url to a pretrained model archive containing:
                . `bert_config.json` a configuration file for the model
                . `model.chkpt` a TensorFlow checkpoint
        from_tf: should we load the weights from a locally saved TensorFlow checkpoint
        cache_dir: an optional path to a folder in which the pre-trained models will be cached.
        state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
        *inputs, **kwargs: additional input for the specific Bert class
            (ex: num_labels for BertForSequenceClassification)
    """
    model = BertForMaskedLM.from_pretrained(*args, **kwargs)
    return model


#def bertForSequenceClassification(*args, **kwargs):
#    model = BertForSequenceClassification.from_pretrained(*args, **kwargs)
#    return model


#def bertForMultipleChoice(*args, **kwargs):
#    model = BertForMultipleChoice.from_pretrained(*args, **kwargs)
#    return model


def bertForQuestionAnswering(*args, **kwargs):
    """
    BertForQuestionAnswering is a fine-tuning model that includes BertModel with
    a token-level classifiers on top of the full sequence of last hidden states.
    Params:
        pretrained_model_name_or_path: either:
            - a str with the name of a pre-trained model to load selected in the list of:
                . `bert-base-uncased`
                . `bert-large-uncased`
                . `bert-base-cased`
                . `bert-large-cased`
                . `bert-base-multilingual-uncased`
                . `bert-base-multilingual-cased`
                . `bert-base-chinese`
            - a path or url to a pretrained model archive containing:
                . `bert_config.json` a configuration file for the model
                . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
            - a path or url to a pretrained model archive containing:
                . `bert_config.json` a configuration file for the model
                . `model.chkpt` a TensorFlow checkpoint
        from_tf: should we load the weights from a locally saved TensorFlow checkpoint
        cache_dir: an optional path to a folder in which the pre-trained models will be cached.
        state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
        *inputs, **kwargs: additional input for the specific Bert class
            (ex: num_labels for BertForSequenceClassification)
    """
    model = BertForQuestionAnswering.from_pretrained(*args, **kwargs)
    return model



