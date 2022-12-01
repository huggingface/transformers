

import torch

from ..bert.modeling_bert import BertModel
from ...trainer import Trainer
from ...modeling_utils import PreTrainedModel
from ..t5.modeling_t5 import T5ForConditionalGeneration, T5Stack

class AtlasConfig():
    pass

class AtlasTrainer(Trainer):
    pass

class AtlasPreTrainedModel(PreTrainedModel):
    pass

class AtlasModel(AtlasPreTrainedModel):
    def __init__(self, questionPassageEncoder, reader, retriever):
        self.questionPassageEncoder = questionPassageEncoder # UntiedDualEncoder
        self.reader = reader # FiD
        self.retriever = retriever # HFIndexBase

class FiD(T5ForConditionalGeneration):
    def __init__(self):
        self.encoder = FiDStack()
        self.decoder = FiDStack()

class FiDStack(T5Stack):
    pass

class UntiedDualEncoder(torch.nn.Module):
    def __init__(self, query_contriever, passage_contriever):
        self.query_contriever = query_contriever
        self.passage_contriever = passage_contriever

class Contriever(BertModel):
    pass

# Copy of RAG model's HFIndexBase class
class HFIndexBase():
    pass

class AtlasRetriever:
    def __init__(self, index):
        self.index = index # HFIndexBase