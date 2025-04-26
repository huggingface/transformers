# This file is only imported if torch is not available.
from ...utils import DummyObject

HindiCausalLMModel = DummyObject("HindiCausalLMModel", ["torch"])
HindiCausalLMForCausalLM = DummyObject("HindiCausalLMForCausalLM", ["torch"])
HindiCausalLMPreTrainedModel = DummyObject("HindiCausalLMPreTrainedModel", ["torch"])
