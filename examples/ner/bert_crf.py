import logging
from transformers.modeling_bert import BertModel, BertPreTrainedModel
from transformers.configuration_bert import BertConfig

from examples.ner.crf import *
from examples.ner.utils_ner_crf import to_crf_pad, unpad_crf

logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    'bert-base-german-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    'bert-large-cased-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    'bert-large-uncased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-large-cased-whole-word-masking-finetuned-squad': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    'bert-base-cased-finetuned-mrpc': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    'bert-base-german-dbmdz-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-cased-pytorch_model.bin",
    'bert-base-german-dbmdz-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-dbmdz-uncased-pytorch_model.bin",
    'bert-base-japanese': "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-pytorch_model.bin",
    'bert-base-japanese-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-whole-word-masking-pytorch_model.bin",
    'bert-base-japanese-char': "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-pytorch_model.bin",
    'bert-base-japanese-char-whole-word-masking': "https://s3.amazonaws.com/models.huggingface.co/bert/cl-tohoku/bert-base-japanese-char-whole-word-masking-pytorch_model.bin",
    'bert-base-finnish-cased-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-cased-v1/pytorch_model.bin",
    'bert-base-finnish-uncased-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/TurkuNLP/bert-base-finnish-uncased-v1/pytorch_model.bin",
}


class BertCRFForTokenClassification(BertPreTrainedModel):
    # def __init__(self, data_parallel=True):
    #     bert = BertModel.from_pretrained("bert-base-cased").to(device=torch.device("cuda"))
    #     if data_parallel:
    #         self.bert = torch.nn.DataParallel(bert)
    #     else:
    #         self.bert = bert
    #     bert_dim = 786  # (or get the dim from BertEmbeddings)
    #     n_labels = 5  # need to set this for your task
    #     self.out = torch.nn.Linear(bert_dim, n_labels)
    #     ...  # droput, log_softmax...
    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "bert"

    def __init__(self, config):
        super(BertCRFForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels + 2)
        self.crf = CRF(self.num_labels)

        self.init_weights()

    def _get_features(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        feats = self.classifier(sequence_output)
        return feats, outputs

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, pad_token_label_id=None):

        logits, outputs = self._get_features(input_ids, attention_mask, token_type_ids,
                                             position_ids, head_mask, inputs_embeds)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            # loss_fct = nn.CrossEntropyLoss()
            pad_mask = (labels != pad_token_label_id)

            # Only keep active parts of the loss
            if attention_mask is not None:
                # active_loss = attention_mask.view(-1) == 1
                # active_logits = logits.view(-1, self.num_labels)[active_loss]
                # active_labels = labels.view(-1)[active_loss]
                loss_mask = ((attention_mask == 1) & pad_mask)
            else:
                # loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
                loss_mask = ((torch.ones(logits.shape) == 1) & pad_mask)

            crf_labels, crf_mask = to_crf_pad(labels, loss_mask, pad_token_label_id)
            crf_logits, _ = to_crf_pad(logits, loss_mask, pad_token_label_id)

            loss = self.crf.neg_log_likelihood(crf_logits, crf_mask, crf_labels)
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            best_path = self.crf(crf_logits, crf_mask)  # (torch.ones(logits.shape) == 1)
            best_path = unpad_crf(best_path, crf_mask, labels, pad_mask)
            outputs = (loss,) + outputs + (best_path,)
        else:
            # removing mask stuff from the output path is done later in my_crf_ner but it should be kept away
            # when calculating loss
            if attention_mask is not None:
                mask = (attention_mask == 1)  # & (labels!=-100))
            else:
                mask = torch.ones(logits.shape).bool()  # (labels!=-100)
            crf_logits, crf_mask = to_crf_pad(logits, mask, pad_token_label_id)
            crf_mask = crf_mask.sum(axis=2) == crf_mask.shape[2]
            best_path = self.crf(crf_logits, crf_mask)
            temp_labels = torch.ones(mask.shape) * pad_token_label_id
            best_path = unpad_crf(best_path, crf_mask, temp_labels, mask)
            outputs = outputs + (best_path,)

        return outputs



