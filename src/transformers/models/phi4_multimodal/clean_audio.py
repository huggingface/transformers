# ruff: noqa
import math
from typing import Callable, List, Optional

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn

from ...configuration_utils import PretrainedConfig


logger = logging.get_logger(__name__)


class Phi4AudioConfig(PretrainedConfig):
    def __init__(
        self,
        hidden_size: int = 1024,  # attention_dim
        num_attention_heads: int = 16,  # attention_heads
        intermediate_size: int = 2048,
        activation: str = "swish",
        chunk_size: int = None,
        left_chunk: int = None,
        num_lang: int = None,
        num_blocks: int = 6,
        dropout_rate: float = 0.0,
        input_layer: str = "nemo_conv",
        causal: bool = True,
        batch_norm: bool = False,
        ext_pw_out_channel: int = 0,
        ext_pw_kernel_size: int = 1,
        depthwise_seperable_out_channel: int = 256,
        depthwise_multiplier: int = 1,
        chunk_se: int = 0,
        kernel_size: int = 3,
        conv_activation: str = "relu",
        conv_glu_type: str = "sigmoid",
        bias_in_glu: bool = True,
        linear_glu_in_convm: bool = False,
        attention_glu_type: str = "swish",
        extra_layer_output_idx: int = -1,
        extra_multi_layer_output_idxs: list = [],
        activation_checkpointing: str = "",
        relative_attention_bias_args: dict = None,
        time_reduction: int = 4,
        replication_pad_for_subsample_embedding: bool = False,
        attention_group_size: int = 1,
        encoder_embedding_config: dict = None,
        positional_dropout_rate: float = 0.0,
    ):
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.activation = activation
        self.chunk_size = chunk_size
        self.left_chunk = left_chunk
        self.num_lang = num_lang
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.input_layer = input_layer
        self.causal = causal
        self.batch_norm = batch_norm
        self.ext_pw_out_channel = ext_pw_out_channel
        self.ext_pw_kernel_size = ext_pw_kernel_size
        self.depthwise_seperable_out_channel = depthwise_seperable_out_channel
        self.depthwise_multiplier = depthwise_multiplier
        self.chunk_se = chunk_se
        self.kernel_size = kernel_size
        self.conv_activation = conv_activation
        self.conv_glu_type = conv_glu_type
        self.bias_in_glu = bias_in_glu
        self.linear_glu_in_convm = linear_glu_in_convm
        self.attention_glu_type = attention_glu_type
        self.extra_layer_output_idx = extra_layer_output_idx
        self.extra_multi_layer_output_idxs = extra_multi_layer_output_idxs
        self.activation_checkpointing = activation_checkpointing
        self.relative_attention_bias_args = relative_attention_bias_args
        self.time_reduction = time_reduction
        self.replication_pad_for_subsample_embedding = replication_pad_for_subsample_embedding
        self.attention_group_size = attention_group_size
        self.encoder_embedding_config = encoder_embedding_config
        self.positional_dropout_rate = positional_dropout_rate

        self.nemo_conv_settings = {
            "subsampling": "dw_striding",
            "subsampling_factor": self.time_reduction,
            "conv_channels": 1024,
            "activation": "relu",
            "is_causal": False,
        }
        self.encoder_embedding_config = {
            "input_size": 80,
        }


class Phi4AudioMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.act_fn = ACT2FN[config.bias_in_glu]
        # ALL AFTER THIS WAS INSIDE A nn.Sequntial CALLED `net` -> KEY CONVERSION
        # gate_up_proj was additionally inside a GLULinear module with `linear` name inside
        self.gate_up_proj = nn.Linear(config.hidden_size, config.intermediate_size * 2, config.bias_in_glu)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = (nn.Dropout(config.dropout_rate),)

    def forward(self, hidden_states):
        up_states = self.gate_up_proj(hidden_states)
        up_states, gate = up_states.chunk(2, dim=-1)
        up_states = up_states * self.act_fn(gate)
        up_states = self.dropout(up_states)
        hidden_states = self.down_proj(up_states)
        out = self.dropout(out)

        return out


def audio_eager_attention_forward(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


class Phi4AudioAttention(nn.Module):
    def __init__(self, config):
        self.config = config
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.dropout_rate
        self.is_causal = True

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dim, bias=True)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dim, config.hidden_size, bias=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mask: Optional[torch.Tensor],
        relative_attention_bias: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        attention_mask = None
        if mask is not None:
            mask = mask.unsqueeze(1)
            if relative_attention_bias is not None:
                attention_mask = mask + relative_attention_bias
            else:
                attention_mask = mask

        attention_interface: Callable = audio_eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class Phi4AudioDepthWiseSeperableConv1d(nn.Module):
    def __init__(self, config, padding=0):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size * config.depthwise_multiplier,
            config.kernel_size,
            1,
            padding=padding,
            groups=config.hidden_size,
        )
        self.pw_conv = nn.Conv1d(
            config.hidden_size * config.depthwise_multiplier, config.depthwise_seperable_out_channel, 1, 1, 0
        )

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class Phi4AudioGluPointWiseConv(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.output_dim = config.ext_pw_out_channel
        kernel_size = config.ext_pw_kernel_size

        self.ext_pw_conv_1d = nn.Conv1d(
            config.hidden_size,
            config.ext_pw_out_channel * 2,
            kernel_size,
            1,
            padding=(kernel_size - 1) if config.causal else (kernel_size - 1) // 2,
        )

        if config.glu_type == "sigmoid":
            self.glu_act = nn.Sigmoid()
        elif config.glu_type == "relu":
            self.glu_act = nn.ReLU()
        elif config.glu_type == "gelu":
            self.glu_act = nn.GELU()
        elif config.glu_type == "swish":
            self.glu_act = Swish()
        else:
            raise ValueError(f"Unsupported activation type {self.glu_act}")

        if config.bias_in_glu:
            self.b1 = nn.Parameter(torch.zeros(1, config.ext_pw_out_channel, 1))
            self.b2 = nn.Parameter(torch.zeros(1, config.ext_pw_out_channel, 1))

    def forward(self, x):
        """
        Args:
            x: torch.Tensor
                input tensor
        """
        # to be consistent with GLULinear, we assume the input always has the #channel (#dim) in the last dimension of the tensor, so need to switch the dimension first for 1D-Conv case
        x = x.permute([0, 2, 1])
        x = self.ext_pw_conv_1d(x)
        if self.bias_in_glu:
            x = (x[:, 0 : self.output_dim, :] + self.b1) * self.glu_act(
                x[:, self.output_dim : self.output_dim * 2, :] + self.b2
            )
        else:
            x = (x[:, 0 : self.output_dim, :]) * self.glu_act(x[:, self.output_dim : self.output_dim * 2, :])

        x = x.permute([0, 2, 1])
        return x


class Phi4AudioConvModule(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.batch_norm = config.batch_norm
        self.kernel_size = config.kernel_size

        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.glu = Phi4AudioGluPointWiseConv(config)
        self.ln1 = (
            nn.Linear(config.ext_pw_out_channel, config.hidden_size)
            if config.hidden_size != config.ext_pw_out_channel
            else nn.Identity()
        )

        if config.causal and config.export:
            padding = 0
        elif config.causal:
            padding = config.kernel_size - 1
        else:
            padding = (config.kernel_size - 1) // 2
        self.dw_sep_conv_1d = Phi4AudioDepthWiseSeperableConv1d(config, padding=padding)

        if config.hidden_size != config.depthwise_seperable_out_channel:
            self.ln2 = nn.Linear(config.depthwise_seperable_out_channel, config.hidden_size)
        if config.batch_norm:
            self.bn_layer = nn.BatchNorm1d(config.hidden_size)

        self.act = ACT2FN[config.activation]

        self.ext_pw_conv_1d = nn.Conv1d(
            config.hidden_size,
            config.ext_pw_out_channel,
            config.ext_pw_kernel_size,
            1,
            padding=config.ext_pw_kernel_size - 1 if config.causal else (config.ext_pw_kernel_size - 1) // 2,
        )
        self.fix_len1 = True if config.causal and config.ext_pw_kernel_size > 1 else False
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x):
        """ConvModule Forward.

        Args:
            x: torch.Tensor
                input tensor.
        """
        x = self.layer_norm(x)

        x = self.glu(x)
        if self.causal and self.ext_pw_kernel_size > 1:
            x = x[:, : -(self.ext_pw_kernel_size - 1), :]
        x = self.ln1(x)

        x = x.permute([0, 2, 1])

        x = self.dw_sep_conv_1d(x)
        if self.causal and self.kernel_size > 1:
            x = x[:, :, : -(self.kernel_size - 1)]
        if hasattr(self, "ln2"):
            x = x.permute([0, 2, 1])
            x = self.ln2(x)
            x = x.permute([0, 2, 1])
        if self.batch_norm:
            x = self.bn_layer(x)
        x = self.act(x)

        x = self.ext_pw_conv_1d(x)
        if self.fix_len1:
            x = x[:, :, : -(self.ext_pw_kernel_size - 1)]

        if self.apply_ln1:
            x = x.permute([0, 2, 1])
            x = self.ln1(x)
            x = x.permute([0, 2, 1])

        x = x.permute([0, 2, 1])
        x = self.dropout(x)
        return x


class Phi4AudioConformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.feed_forward_in = Phi4AudioMLP(config)
        self.self_attn = Phi4AudioAttention(config)
        self.conv = Phi4AudioConvModule(config)
        self.feed_forward_out = Phi4AudioMLP(config)
        self.layer_norm_att = nn.LayerNorm(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

    def forward(
        self,
        x,
        pos_k,
        pos_v,
        mask,
        relative_attention_bias: Optional[torch.Tensor] = None,
    ):
        """ConformerEncoder forward.

        Args:
            x: torch.Tensor
                input feature of shape (batch, max_time_in, size)
            pos_k: torch.Tensor
                positional key embedding.
            mask: torch.Tensor
                mask for x (batch, max_time_in)
            relative_attention_bias: Optional[torch.Tensor]
                bias added to attention logits w.r.t. relative positions (1, n_head, time1, time2)
        """
        x = x + 0.5 * self.feed_forward_in(x)
        norm_x = self.layer_norm_att(x)

        x = x + self.self_attn(
            norm_x,
            pos_k,
            pos_v,
            mask,
            relative_attention_bias=relative_attention_bias,
        )
        x = x + self.conv(x)
        x = x + 0.5 * self.feed_forward_out(x)

        out = self.layer_norm(x)

        return out


class Phi4AudioNemoConvSubsampling(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.subsampling_factor = self.config.nemo_conv_settings["subsampling_factor"]

        if self.subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self.sampling_num = int(math.log(self.subsampling_factor, 2))

        self.act_fn = ACT2CLS[self.config.nemo_conv_settings["activation"]]

        conv_channels = self.config.nemo_conv_settings["conv_channels"]
        layers = []

        layers.append(
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=conv_channels,
                kernel_size=3,
                stride=2,
                padding=1,
            )
        )
        layers.append(self.act_fn)

        for _ in range(self.sampling_num - 1):
            layers.append(
                torch.nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    groups=conv_channels,
                )
            )
            layers.append(
                torch.nn.Conv2d(
                    in_channels=conv_channels,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    groups=1,
                )
            )
            layers.append(self.act_fn)

        self.conv = torch.nn.Sequential(*layers)

        in_length = torch.tensor(config.hidden_size, dtype=torch.float)
        out_length = calc_length(
            lengths=in_length,
            all_paddings=2,
            kernel_size=3,
            stride=2,
            ceil_mode=False,
            repeat_num=self.sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), config.hidden_size)

    def forward(self, x, mask):
        """
        Forward method for NeMo subsampling.

        Args:
            x[Batch, Time, Filters]: torch.Tensor
                input tensor
            x_mask: torch.Tensor
                input mask

        Returns:
            x: torch.Tensor
                Resulting tensor from subsampling (B, T // time_reduction_factor, feat_out)
            pad_mask: torch.Tensor
                tensor of padded hidden state sequences (B, 1, T // time_reduction_factor)
        """
        # Unsqueeze Channel Axis
        x = x.unsqueeze(1)

        x = self.conv(x)

        # Flatten Channel and Frequency Axes
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))

        if mask is None:
            return x, None

        max_audio_length = x.shape[1]
        feature_lens = mask.sum(1)
        padding_length = torch.ceil(feature_lens / self.subsampling_factor)
        pad_mask = torch.arange(0, max_audio_length, device=x.device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(1)
        return x, pad_mask.unsqueeze(1)


def calc_length(lengths, all_paddings, kernel_size, stride, ceil_mode, repeat_num=1):
    """Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


class Phi4AudioRelativeAttentionLogitBias(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.num_heads = config.num_attention_heads
        self.max_distance = config.relative_attention_bias_args.get("t5_bias_max_distance", 1000)
        self.symmetric = config.relative_attention_bias_args.get("t5_bias_symmetric", False)
        self.num_buckets = self.max_distance
        if not self.symmetric:
            self.num_buckets *= 2
        self.bias_values = nn.Embedding(self.num_buckets, self.num_heads)

    def forward(self, x):
        # instantiate bias compatible with shape of x
        max_pos = x.size(1)
        context_position = torch.arange(max_pos, device=x.device, dtype=torch.long)[:, None]
        memory_position = torch.arange(max_pos, device=x.device, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position
        # clipping to a maximum distance using ops that play well with ONNX export
        relative_position = relative_position.masked_fill(relative_position < -self.max_distance, -self.max_distance)
        relative_position = relative_position.masked_fill(
            relative_position > self.max_distance - 1, self.max_distance - 1
        )

        # mapping from relative position to index in the bias parameter
        bias_idx = relative_position
        if self.symmetric:
            bias_idx = bias_idx.abs()
        else:
            bias_idx += self.num_buckets // 2

        t5_rel_att_bias = self.bias_values(bias_idx)  # [L, L, H]
        t5_rel_att_bias = t5_rel_att_bias.permute(2, 0, 1).unsqueeze(0)  # [1, H, L, L]

        return t5_rel_att_bias


class Phi4AudioMeanVarianceNormLayer(nn.Module):
    """Mean/variance normalization layer.

    Will substract mean and multiply input by inverted standard deviation.
    Typically used as a very first layer in a model.

    Args:
        input_size: int
            layer input size.
    """

    def __init__(self, config):
        super().__init__()
        self.register_buffer("global_mean", torch.zeros(config.encoder_embedding_config["input_size"]))
        self.register_buffer("global_invstd", torch.ones(config.encoder_embedding_config["input_size"]))

    def forward(self, x):
        return (x - self.global_mean) * self.global_invstd


class Phi4AudioConformerEncoder(nn.Module):
    extra_multi_layer_output_idxs: List[int]

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder_embedding = Phi4AudioMeanVarianceNormLayer(config)
        self.embed = Phi4AudioNemoConvSubsampling(config)
        self.relative_attention_bias_layer = Phi4AudioRelativeAttentionLogitBias(config)
        self.encoders = nn.ModuleList([Phi4AudioConformerEncoderLayer(config) for _ in range(config.num_blocks)])

    def _chunk_size_selection(self, chunk_size=None, left_chunk=None):
        """If chunk size is a list, we will randomly select a chunk size."""

        if chunk_size is None:
            chunk_size = self.chunk_size
        if left_chunk is None:
            left_chunk = self.left_chunk
        if isinstance(chunk_size, list):
            # Variable chunk size during training
            chunk_size_index = int(torch.randint(low=0, high=len(chunk_size), size=(1,)))
            chunk_size_train_eff = chunk_size[chunk_size_index]
            if not isinstance(left_chunk, list):
                raise ValueError("Since chunk_size is a list, left_chunk must be a list")
            if len(left_chunk) != len(chunk_size):
                raise ValueError("The length of left_chunk must be the same as length of chunk_size.")
            left_chunk_train_eff = left_chunk[chunk_size_index]
        else:
            chunk_size_train_eff = chunk_size
            left_chunk_train_eff = left_chunk

        return chunk_size_train_eff, left_chunk_train_eff

    def _streaming_mask(self, seq_len, batch_size, chunk_size, left_chunk):
        chunk_size_train_eff, left_chunk_train_eff = self._chunk_size_selection(chunk_size, left_chunk)

        # Create mask matrix for streaming
        # S stores start index. if chunksize is 18, s is [0,18,36,....]
        chunk_start_idx = np.arange(0, seq_len, chunk_size_train_eff)
        # avoid randomness when run evaluation or decoding
        if self.training and np.random.rand() > 0.5:
            # Either first or last chunk is not complete.
            # If only the last one is not complete, EOS is not effective
            chunk_start_idx = seq_len - chunk_start_idx
            chunk_start_idx = chunk_start_idx[::-1]
            chunk_start_idx = chunk_start_idx[:-1]
            chunk_start_idx = np.insert(chunk_start_idx, 0, 0)

        enc_streaming_mask = (
            adaptive_enc_mask(seq_len, chunk_start_idx, left_window=left_chunk_train_eff)
            .unsqueeze(0)
            .expand([batch_size, -1, -1])
        )
        return enc_streaming_mask

    def forward_embeddings(self, xs_pad, masks):
        """Forwarding the inputs through the top embedding layers

        Args:
            xs_pad: torch.Tensor
                input tensor
            masks: torch.Tensor
                input mask
        """
        # pylint: disable=R0915
        # get new lens.
        seq_len = math.ceil(xs_pad.shape[1] / self.config.time_reduction)
        if seq_len <= 0:
            raise ValueError(
                f"""The squence length after time reduction is invalid: {seq_len}.
                Your input feature is too short. Consider filtering out the very
                short sentence from data loader""",
            )

        batch_size = xs_pad.shape[0]

        enc_streaming_mask = self._streaming_mask(seq_len, batch_size, self.config.chunk_size, self.config.left_chunk)

        if xs_pad.is_cuda:
            enc_streaming_mask = enc_streaming_mask.cuda()
            xs_pad = xs_pad.cuda()

        input_tensor = xs_pad
        input_tensor, masks = self.embed(input_tensor, masks)

        streaming_mask = enc_streaming_mask
        if streaming_mask is not None and masks is not None:
            hs_mask = masks & streaming_mask
        elif masks is not None:
            hs_mask = masks
        else:
            hs_mask = streaming_mask

        return input_tensor, hs_mask, masks

    def calculate_hs_mask(self, xs_pad, device, mask):
        max_audio_length = xs_pad.shape[1]
        batch_size = xs_pad.shape[0]
        enc_streaming_mask = self._streaming_mask(max_audio_length, batch_size, self.chunk_size, self.left_chunk)
        enc_streaming_mask = enc_streaming_mask.to(device)
        if mask is None:
            return enc_streaming_mask

        feature_lens = mask.sum(1)
        padding_length = feature_lens
        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(1)
        pad_mask = pad_mask.unsqueeze(1)
        pad_mask = pad_mask & enc_streaming_mask
        return pad_mask

    def forward(self, xs_pad, masks):
        """Conformer Forward function

        Args:
            xs_pad: torch.Tensor
                input tensor
            masks: torch.Tensor
                post-embedding input lengths
        """
        xs_pad = self.encoder_embedding(xs_pad)
        input_tensor, hs_mask, masks = self.forward_embeddings(xs_pad, masks)

        unfolded = False
        ori_bz, seq_len, D = input_tensor.shape
        max_seq_len = 500  # maxium position for absolute positional encoding
        if seq_len > max_seq_len:
            # audio sequence is longer than max_seq_len, unfold it into chunks of max_seq_len
            unfolded = True
            # the unfold op will drop residual frames, pad it to the multiple of max_seq_len
            if seq_len % max_seq_len > 0:
                chunk_pad_size = max_seq_len - (seq_len % max_seq_len)
            else:
                chunk_pad_size = 0
            if chunk_pad_size > 0:
                input_tensor_pad = F.pad(input_tensor, (0, 0, 0, chunk_pad_size), "constant", 0)
                input_tensor = input_tensor_pad.to(input_tensor.device)

            input_tensor = unfold_tensor(input_tensor, max_seq_len)
            if masks is not None:
                # revise hs_mask here because the previous calculated hs_mask did not consider extra pad
                subsampled_pad_mask = masks.squeeze(1)  # [bz, subsampled_unmask_seq_len]
                extra_padded_subsamlped_pad_mask = F.pad(
                    subsampled_pad_mask, (0, chunk_pad_size), "constant", False
                )  # extra padding to the pad mask
                extra_padded_subsamlped_pad_mask = extra_padded_subsamlped_pad_mask.unsqueeze(-1).float()
                masks_unfold = unfold_tensor(
                    extra_padded_subsamlped_pad_mask, max_seq_len
                )  # unfold the pad mask like we did to the input tensor
                masks_unfold = masks_unfold.squeeze(-1).bool()  # unfold op does not support bool tensor
            else:
                masks_unfold = None
            hs_mask = self.calculate_hs_mask(
                input_tensor, input_tensor.device, masks_unfold
            )  # calculate hs_mask based on the unfolded pad mask

        relative_attention_bias = self.relative_attention_bias_layer(input_tensor)

        for layer in self.encoders:
            input_tensor = layer(
                input_tensor,
                hs_mask,
                relative_attention_bias=relative_attention_bias,
            )

        if unfolded:
            embed_dim = input_tensor.shape[-1]
            input_tensor = input_tensor.reshape(ori_bz, -1, embed_dim)
            # if we ever padded before unfolding, we need to remove the padding
            if chunk_pad_size > 0:
                input_tensor = input_tensor[:, :-chunk_pad_size, :]

        return input_tensor, masks


def unfold_tensor(xs_pad, max_seq_len):
    """
    For a given tensor with shape of (N, T, D), if sequence length T is longer than max_seq_len,
    this function unfold it to a (NT', max_seq_len, D) where T' is T // max_seq_len.
    Args:
        xs_pad: N, T, D
    """
    _, _, D = xs_pad.shape
    xs_pad = xs_pad.transpose(-1, -2)  # convert to N, D, T
    # N x D x 1 x T => N x (D x max_seq_len) x T'
    xs_pad = F.unfold(
        xs_pad[..., None, :],
        kernel_size=(1, max_seq_len),
        stride=(1, max_seq_len),
    )

    new_bsz, _, slen = xs_pad.shape
    # N x D x max_seq_len x T'
    xs_pad = xs_pad.view(new_bsz, -1, max_seq_len, slen)
    # N x T' x max_seq_len x D
    xs_pad = xs_pad.permute(0, 3, 2, 1).contiguous()
    # NT' x max_seq_len x D
    xs_pad = xs_pad.view(-1, max_seq_len, D)
    return xs_pad


def adaptive_enc_mask(x_len, chunk_start_idx, left_window=0, right_window=0):
    """
    The function is very important for Transformer Transducer Streaming mode
    Args:
        xs_len (int): sequence length
        chunk_start_idx (list): first idx of each chunk, such as [0,18,36,48]. It also supports adaptive chunk size [0,10,15,45]
        left_window (int): how many left chunks can be seen
        right_window (int): how many right chunks can be seen. It is used for chunk overlap model.
        Returns:
            mask (torch.Tensor): a mask tensor for streaming model
            Torch 1.0.1
            tensor([[1., 1., 0., 0.],
                    [0., 1., 1., 0.],
                    [0., 0., 1., 1.]])
            Torch 1.4.1
            tensor([[True., True., False., False.],
                    [False., True., True., False.],
                    [False., False., True., True.]])
    """
    chunk_start_idx = torch.Tensor(chunk_start_idx).long()  # first idx of each chunk, such as [0,18,36,48].
    start_pad = torch.nn.functional.pad(
        chunk_start_idx, (1, 0)
    )  # append 0 to the beginning, so it becomes [0, 0, 18, 36, 48]
    end_pad = torch.nn.functional.pad(
        chunk_start_idx, (0, 1), value=x_len
    )  # append x_len to the end, so it becomes [0,18,36,48, x_len]
    seq_range = torch.arange(0, x_len).unsqueeze(-1)  # seq_range size: [x_len, 1]
    idx = ((seq_range < end_pad) & (seq_range >= start_pad)).nonzero()[:, 1]  # idx size: [x_len]
    boundary = end_pad[idx]  # boundary size: [x_len]
    seq_range_expand = torch.arange(0, x_len).unsqueeze(0).expand(x_len, -1)  # seq_range_expand size [x_len, x_len]
    idx_left = idx - left_window
    idx_left[idx_left < 0] = 0
    boundary_left = start_pad[idx_left]
    mask_left = seq_range_expand >= boundary_left.unsqueeze(-1)
    idx_right = idx + right_window
    idx_right[idx_right > len(chunk_start_idx)] = len(chunk_start_idx)
    boundary_right = end_pad[idx_right]
    mask_right = seq_range_expand < boundary_right.unsqueeze(-1)
    return mask_left & mask_right
