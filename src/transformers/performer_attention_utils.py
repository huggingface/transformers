from .configuration_performer_attention import PerformerAttentionConfig
from .modeling_performer_attention import PerformerAttention

from copy import copy, deepcopy
from functools import wraps


def supports_performer_attention(class_to_process):
    """
    Decorator for model config classes that adds the attributes 'attention_type: str' and
    'performer_attention_config: Optional[Union[dict, PerformerAttentionConfig]]', along with synthesizing a bit of
    boilerplate code to make sure JSON serialization works properly.
    :param class_to_process: The model config class to process.
    :return: The same class object that was passed as class_to_process, but now with new attributes.

    Examples::
    >>> @supports_performer_attention
    >>> class BertConfig(PretrainedConfig):
    """
    unwrapped_init = class_to_process.__init__

    @wraps(unwrapped_init)
    def wrapped_init(self, **kwargs):
        self.attention_type = kwargs.pop("attention_type", "softmax")
        performer_attention_config = kwargs.pop("performer_attention_config", None)

        if isinstance(performer_attention_config, dict):
            self.performer_attention_config = PerformerAttentionConfig(**performer_attention_config)
        else:
            self.performer_attention_config = performer_attention_config

        unwrapped_init(self, **kwargs)

    def new_to_dict(self):
        output = super(class_to_process, self).to_dict()

        # Correct for the fact that PretrainedConfig doesn't call .__dict__ recursively on non-JSON primitives
        performer_config = output['performer_attention_config']
        if performer_config is not None:
            output['performer_attention_config'] = deepcopy(performer_config.to_dict())

        return output

    class_to_process.__init__ = wrapped_init
    class_to_process.to_dict = new_to_dict

    return class_to_process


def init_performer_attention(softmax_attention_class, attention_attribute_name: str = 'attention', **kwargs):
    """
    Decorator for __init__ methods on transformer layers, synthesizing boilerplate code to initialize the
    PerformerAttention object based on the model config parameters if necessary.
    :param softmax_attention_class: The class object of the attention module to use when attention_type == 'softmax'.
    :param attention_attribute_name: The name of the attribute in which to store the PerformerAttention object.
    :param kwargs: The keys are PerformerAttentionConfig parameter names. The values are either strings
        representing attributes in the model config class— in which case the PerformerAttentionConfig object will
        copy values from the model config object— or they are arbitrary objects which will be directly bound to the
        parameters. Commonly 'linear_layer_names' is included in kwargs to make sure the PerformerAttention object will
        imitate the naming convention of softmax_attention_class.
    :return: The wrapped __init__ method

    Examples::
    >>> class TransformerBlock(nn.Module):
    >>>     @init_performer_attention(softmax_attention_class=MultiHeadSelfAttention,
    >>>                               linear_layer_names=('q_lin', 'k_lin', 'v_lin', 'out_lin'),
    >>>                               d_model='dim', num_heads='n_heads')
    >>>     def __init__(self, config):
    >>>         super().__init__()
    >>>         # Other initialization code here...
    """

    def wrapper(unwrapped_init):
        @wraps(unwrapped_init)
        def wrapped_init(self, config):
            unwrapped_init(self, config)    # This should call super()

            attn_type = config.attention_type
            if attn_type == 'softmax':
                setattr(self, attention_attribute_name, softmax_attention_class(config))

            elif attn_type == 'performer':
                attn_config = config.performer_attention_config or PerformerAttentionConfig()
                kwarg_copy = copy(kwargs)

                attn_config.attention_dropout = getattr(
                    config,
                    kwarg_copy.pop('attention_dropout', 'attention_dropout'),
                    0.0
                )
                attn_config.d_model = getattr(config, kwarg_copy.pop('d_model', 'd_model'))
                attn_config.num_heads = getattr(config, kwarg_copy.pop('num_heads', 'num_heads'))
                attn_config.__dict__.update(kwarg_copy)    # Apply any remaining kwargs directly

                setattr(self, attention_attribute_name, PerformerAttention(attn_config))
            else:
                raise ValueError(f"Invalid attention_type {attn_type}")

        return wrapped_init

    return wrapper


# For BERT and models copied from BERT
def init_performer_attention_bertlike(softmax_attention_class):
    return init_performer_attention(
        softmax_attention_class=softmax_attention_class, attention_attribute_name='self',
        linear_layer_names=('query', 'key', 'value'), d_model='hidden_size', num_heads='num_attention_heads',
        attention_dropout='attention_probs_dropout_prob'
    )
