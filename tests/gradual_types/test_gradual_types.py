import inspect
import unittest
import torch
from torch.fx.experimental.migrate_gradual_types.transform_to_z3 import transform_all_constraints
from torch.fx.experimental.migrate_gradual_types.z3_types import tensor_type
from torch.fx.passes.shape_prop import ShapeProp
from torch.fx import GraphModule
from enum import Enum
from src.transformers import *
import src.transformers.utils.fx as fx
import z3

bs = 4
num_choices = 3
seq_length = 32


class MultiUseParameterConfig(Enum):
    TRANSMIT = 1
    REPLICATE = 2


hf_tracer = fx.HFTracer()


def generate_concrete_args_for_model(model, input_names=None):
    input_names = input_names if input_names else model.dummy_inputs.keys()
    sig = inspect.signature(model.forward)
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}
    return concrete_args


def generate_hf_model(model_cls):
    config_cls = model_cls.config_class
    config = config_cls()
    # we simplify the model for now by removing the hidden layers
    config.num_hidden_layers = 0
    if model_cls in [GPT2ForSequenceClassification, GPTNeoForSequenceClassification, GPTJForSequenceClassification] or \
            model_cls.__name__.startswith("Roberta") or model_cls.__name__.startswith("Marian"):
        config.pad_token_id = 0
    model = model_cls(config)
    model.eval()

    return model


def generate_inputs_for_model(model_cls, model, include_loss_args=False):
    if model_cls.__name__.endswith('MultipleChoice'):
        input = torch.zeros(bs, num_choices, seq_length, dtype=torch.long).random_(model.config.vocab_size)
    elif model_cls.__name__.startswith("Roberta"):
        input = torch.zeros(bs, seq_length, dtype=torch.long)
    else:
        input = torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size)

    if 'Bart' in model_cls.__name__:
        input[:, -1] = model.config.eos_token_id

    input_dict = {'input_ids': input}

    if model_cls.__name__.startswith("T5") or model_cls.__name__.startswith("M2M100") \
            or model_cls.__name__.startswith("MT5") or model_cls in [BlenderbotModel, BlenderbotSmallModel,
                                                                     BlenderbotForConditionalGeneration,
                                                                     BlenderbotSmallForConditionalGeneration,
                                                                     PegasusModel, PegasusForConditionalGeneration,
                                                                     MarianModel, MarianMTModel]:
        input_dict.update({'decoder_input_ids': input})

    if include_loss_args:
        if model_cls.__name__.endswith('PreTraining'):
            if model_cls == ElectraForPreTraining:
                input_dict.update({
                    'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(1),
                })
            else:
                label_name = 'sentence_order_label' if model_cls in [AlbertForPreTraining] else 'next_sentence_label'
                input_dict.update({
                    'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size),
                    label_name: torch.zeros(bs, dtype=torch.long).random_(1),
                })
        elif model_cls.__name__.endswith('QuestionAnswering'):
            input_dict.update({
                'start_positions': torch.zeros(bs, dtype=torch.long).random_(seq_length),
                'end_positions': torch.zeros(bs, dtype=torch.long).random_(seq_length)
            })
        elif (model_cls.__name__.endswith('MaskedLM') or model_cls.__name__.endswith('HeadModel') or
              model_cls.__name__.endswith('CausalLM') or model_cls.__name__.endswith('DoubleHeadsModel')):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size),
            })
        elif model_cls.__name__.endswith('TokenClassification'):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.num_labels - 1),
            })
        elif model_cls.__name__.endswith('MultipleChoice'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(num_choices),
            })
        elif model_cls.__name__.endswith('SequenceClassification'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(model.config.num_labels - 1),
            })
        elif model_cls.__name__.endswith('NextSentencePrediction'):
            input_dict.update({
                'labels': torch.zeros(bs, dtype=torch.long).random_(1),
            })
        elif model_cls.__name__.endswith('ForConditionalGeneration'):
            input_dict.update({
                'labels': torch.zeros(bs, seq_length, dtype=torch.long).random_(model.config.vocab_size - 1),
            })
        else:
            raise NotImplementedError(f'Class {model_cls.__name__} unsupported for training test ')

    return input_dict


model_classes = [XGLMModel, AlbertModel, BartModel, BertModel, DistilBertModel, ElectraModel, GPT2Model,
 GPTJModel, GPTNeoModel, MegatronBertModel, MobileBertModel, RobertaModel, T5Model,
 BlenderbotModel, BlenderbotSmallModel]

traces = []

# generate traces for HF models using the HF tracer
for c in model_classes:
    m = generate_hf_model(c)
    input_dict = generate_inputs_for_model(c, m)
    replicate = True
    multi_use_param_config = MultiUseParameterConfig.REPLICATE if replicate else MultiUseParameterConfig.TRANSMIT
    concrete_args = generate_concrete_args_for_model(m, input_dict.keys())

    g = GraphModule(m, hf_tracer.trace(m, concrete_args=concrete_args))
    graph = hf_tracer.trace(m, concrete_args=concrete_args)
    g = GraphModule(m, graph)
    traces.append(g)

    # we generate the trace for the first model
    break


class HFModels(unittest.TestCase):

    def test_trace_model(self):
        XGLMModel_trace = traces[0]
        input = torch.ones([2, 4], dtype=torch.long)
        # generate shapes for a particular input to compare with
        # our shape inference
        sample_input = input
        ShapeProp(XGLMModel_trace).propagate(sample_input)

        for n in XGLMModel_trace.graph.nodes:
            if n.target == 'layer_norm':
                layer_norm_size = n.meta['tensor_meta'].shape

        constraints = transform_all_constraints(g, counter=0)
        s = z3.Solver()
        s.add(constraints)
        self.assertEqual(s.check(), z3.sat)
        layer_norm = z3.Const(82, tensor_type)

        # the first dimension is lost due to view
        self.assertEqual(s.model()[layer_norm].arg(0).arg(0), 0)
        self.assertEqual(s.model()[layer_norm].arg(1).arg(1), layer_norm_size[1])
        self.assertEqual(s.model()[layer_norm].arg(2).arg(1), layer_norm_size[2])
