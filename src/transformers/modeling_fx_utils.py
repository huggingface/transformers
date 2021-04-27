import dis
import inspect
from typing import List, Optional

import torch
from torch.fx import Node, Proxy, Tracer

from . import PreTrainedModel


class CustomProxy(Proxy):
    def __init__(self, node: Node, tracer: Optional[Tracer] = None):
        super().__init__(node, tracer=tracer)
        if hasattr(self, "tracer") and self.tracer is not None:
            self.device = self.tracer.root.device
            self.dtype = next(self.tracer.root.parameters()).dtype

    def dim(self):
        return len(self.tracer.encoder_shape)

    def _shape(self, calling_frame):
        module = calling_frame.f_locals.get("self", None)
        is_decoder = hasattr(module, "is_decoder") and module.is_decoder
        return list(self.tracer.decoder_shape) if is_decoder else list(self.tracer.encoder_shape)

    def size(self, dim=None):
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None

        # self.size can be called through the shape property, in which case we need to get the outer
        # frame, containing the meaningful information.
        if calling_frame.f_code.co_name == "shape":
            calling_frame = calling_frame.f_back

        code_context = inspect.getframeinfo(calling_frame).code_context[0].strip()

        shape = self._shape(calling_frame)

        if calling_frame.f_code.co_name == "transpose_for_scores":
            # Provides the proper "x.size()" for:
            # new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
            shape = shape + [-1]
        elif "context_layer" in calling_frame.f_locals:
            # Provides the proper "context_layer.size()" for:
            # new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
            shape = shape + [-1, -1]
        elif calling_frame.f_locals.get("do_cross_attention", False):
            # Provides the proper shape for:
            # query_length = present_key_value_state[0].shape[2]
            # (modeling_t5.py)
            shape = list(self.tracer.encoder_shape)
            shape = shape[:1] + [-1] + shape[1:2]
        elif "key_length" in code_context or "encoder_seq_length" in code_context:
            shape = list(self.tracer.encoder_shape)
        elif "lm_logits.size(-1)" in code_context:
            shape = [self.tracer.root.config.vocab_size]
        elif "start_positions" in code_context or "end_positions" in code_context:
            # For question answering tasks.
            shape = [1]
        elif "num_choices" in code_context:
            if self.tracer.num_choices <= 0:
                raise ValueError("num_choices must be given to the CustomTracer for MultipleChoice tasks.")
            shape = shape[:1] + [self.tracer.num_choices] + shape[1:]
        else:
            instructions = dis.get_instructions(calling_frame.f_code)
            is_unpack = False
            for inst in instructions:
                if inst.opname == "UNPACK_SEQUENCE":
                    is_unpack = True
                    break
            if is_unpack and inst.argval >= 3:
                shape += [self.tracer.root.config.hidden_size]
                dummy_values = [1] * (inst.argval - 3)
                shape += dummy_values

        if dim is not None:
            return shape[dim]

        return tuple(shape)

    @property
    def shape(self):
        return self.size()

    def __bool__(self) -> bool:
        frame = inspect.currentframe()
        assert frame is not None
        calling_frame = frame.f_back
        assert calling_frame is not None
        code_context = inspect.getframeinfo(calling_frame).code_context[0].strip()
        if calling_frame.f_code.co_name == "apply_chunking_to_forward":
            # Returning True to every assertion in "apply_chuncking_to_forward"
            return True
        elif "assert" in code_context:
            return True
        elif calling_frame.f_code.co_name == "get_extended_attention_mask":
            # Corresponding to:
            # if causal_mask.shape[1] < attention_mask.shape[1]:
            return calling_frame.f_back.f_locals["past_key_values"][0] is not None
        raise NotImplementedError("__bool__ was called for CustomProxy, but this case is not covered yet.")

    def __setitem__(self, key, value):
        pass

    def __iter__(self):
        return iter(["dummy"])


class CustomTracer(Tracer):
    def __init__(self, batch_size=1, seqlen=[128, 128], num_choices=-1):
        super().__init__()
        encoder_seqlen = seqlen[0] if isinstance(seqlen, (list, tuple)) else seqlen
        decoder_seqlen = seqlen[1] if isinstance(seqlen, (list, tuple)) else -1
        self.encoder_shape = [batch_size, encoder_seqlen]
        self.decoder_shape = [batch_size, decoder_seqlen] if decoder_seqlen > 0 else list(self.encoder_shape)
        self.num_choices = num_choices
        if self.num_choices > 0:
            self.encoder_shape[0] *= self.num_choices

    def proxy(self, node: Node):
        return CustomProxy(node, self)


def symbolic_trace(
    model: PreTrainedModel, input_names: Optional[List[str]] = None, batch_size=1, seqlen=[128, 128], num_choices=-1
):
    if input_names is None:
        input_names = model.dummy_inputs.keys()

    sig = inspect.signature(model.forward)
    # TODO: how to handle the case of the "return_dict" parameter.
    concrete_args = {p.name: p.default for p in sig.parameters.values() if p.name not in input_names}

    tracer = CustomTracer(batch_size=batch_size, seqlen=seqlen, num_choices=num_choices)
    traced_graph = tracer.trace(model, concrete_args=concrete_args)
    traced = torch.fx.GraphModule(model, traced_graph)

    return traced
