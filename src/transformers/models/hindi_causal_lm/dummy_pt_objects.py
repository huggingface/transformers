from ...utils import DummyObject


class HindiCausalLMPreTrainedModel(metaclass=DummyObject): # <<< Must exist here
    _backends = ["torch"]

class HindiCausalLMModel(metaclass=DummyObject):
    _backends = ["torch"]

class HindiCausalLMForCausalLM(metaclass=DummyObject):
    _backends = ["torch"]
