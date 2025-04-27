# coding=utf-8
from ...utils import DummyObject


class HindiCausalLMPreTrainedModel(metaclass=DummyObject):
    _backends = ["torch"]

class HindiCausalLMModel(metaclass=DummyObject):
    _backends = ["torch"]

class HindiCausalLMForCausalLM(metaclass=DummyObject): # <<< Check this class name carefully!
    _backends = ["torch"]
