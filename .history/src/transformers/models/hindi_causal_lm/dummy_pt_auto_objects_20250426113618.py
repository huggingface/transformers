from ...utils import DummyObject

class HindiCausalLMModel(metaclass=DummyObject):
    _backends = ["torch"]

class HindiCausalLMForCausalLM(metaclass=DummyObject):
    _backends = ["torch"]

class HindiCausalLMPreTrainedModel(metaclass=DummyObject):
    _backends = ["torch"]
