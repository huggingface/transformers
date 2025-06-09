from ...configuration_utils import PretrainedConfig
from ...utils import logging

logger = logging.get_logger(__name__)

class MViTV2Config(PretrainedConfig):
    model_type = "mvitv2"

    def __init__(self,
                 depths=(2, 3, 16, 3),
                 in_channels=3,
                 hidden_size=96,
                 num_heads=1,
                 image_size=(224, 224),
                 patch_kernel_size=(7, 7),
                 patch_stride_size=(4, 4),
                 patch_padding_size=(3, 3),
                 use_cls_token=False,
                 use_absolute_positional_embeddings = False,
                 attention_pool_first=False,
                 expand_feature_dimension_in_attention=True,
                 mode='conv',
                 kernel_qkv=(3, 3),
                 stride_q=((1, 1), (2, 2), (2, 2), (2, 2)),
                 stride_kv_adaptive=(4, 4),
                 stride_kv=None,
                 qkv_bias=True,
                 residual_pooling=True,
                 relative_positional_embeddings_type="spatial",
                 mlp_ratio=4,
                 hidden_activation_function="gelu",
                 drop_path_rate=0.1,
                 drop_rate=0.1,
                 initializer_range=0.02,
                 layer_norm_epsilon=1e-5,
                 **kwargs):
        super().__init__(**kwargs)
        self.depths = depths
        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.image_size = image_size
        self.patch_kernel_size = patch_kernel_size
        self.patch_stride_size = patch_stride_size
        self.patch_padding_size = patch_padding_size
        self.use_cls_token = use_cls_token
        self.use_absolute_positional_embeddings = use_absolute_positional_embeddings

        self.attention_pool_first = attention_pool_first
        self.expand_feature_dimension_in_attention = expand_feature_dimension_in_attention
        self.mode = mode
        self.kernel_qkv = kernel_qkv
        self.stride_q = stride_q
        self.stride_kv_adaptive = stride_kv_adaptive
        self.stride_kv = stride_kv
        self.qkv_bias = qkv_bias
        self.residual_pooling = residual_pooling
        self.relative_positional_embeddings_type = relative_positional_embeddings_type

        self.mlp_ratio = mlp_ratio
        self.hidden_activation_function = hidden_activation_function

        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.initializer_range = initializer_range
        self.layer_norm_epsilon = layer_norm_epsilon
    
    @property
    def classifier_hidden_size(self):
        return self.hidden_size * (2**(len(self.depths) - 1))

__all__ = ["MViTV2Config"]
