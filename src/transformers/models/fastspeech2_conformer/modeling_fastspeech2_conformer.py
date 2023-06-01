"""
Taken from ESPNet and IMS Toucan
"""

import math
from typing import Optional  # , List, Tuple, Union

import numpy
import torch
from torch import nn

from ...modeling_utils import PreTrainedModel
from .configuration_fastspeech2_conformer import FastSpeech2ConformerConfig

def initialize(model, init):
    """
    Initialize weights of a neural network module.

    Parameters are initialized using the given method or distribution.

    Args:
        model: Target.
        init: Method of initialization.
    """

    # weight init
    for p in model.parameters():
        if p.dim() > 1:
            if init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(p.data)
            elif init == "xavier_normal":
                torch.nn.init.xavier_normal_(p.data)
            elif init == "kaiming_uniform":
                torch.nn.init.kaiming_uniform_(p.data, nonlinearity="relu")
            elif init == "kaiming_normal":
                torch.nn.init.kaiming_normal_(p.data, nonlinearity="relu")
            else:
                raise ValueError("Unknown initialization: " + init)
    # bias init
    for p in model.parameters():
        if p.dim() == 1:
            p.data.zero_()

    # reset some modules with default init
    for m in model.modules():
        if isinstance(m, (torch.nn.Embedding, torch.nn.LayerNorm)):
            m.reset_parameters()


def pad_list(xs, pad_value):
    """
    Perform padding for the list of tensors.

    Args:
        xs (List): List of Tensors [(T_1, `*`), (T_2, `*`), ..., (T_B, `*`)].
        pad_value (float): Value for padding.

    Returns:
        Tensor: Padded tensor (batch_size, max_text_length, `*`).

    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def make_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """
    Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (batch_size,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    if not isinstance(lengths, list):
        lengths = lengths.tolist()
    bs = int(len(lengths))
    if xs is None:
        maxlen = int(max(lengths))
    else:
        maxlen = xs.size(length_dim)

    if device is not None:
        seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=device)
    else:
        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(slice(None) if i in (0, length_dim) else None for i in range(xs.dim()))
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1, device=None):
    """
    Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (batch_size,).
        xs (Tensor, optional): The reference tensor.
            If set, masks will be the same shape as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor.
            See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)

    """
    return ~make_pad_mask(lengths, xs, length_dim, device=device)


class DurationPredictor(torch.nn.Module):
    """
    Duration predictor module.

    This is a module of duration predictor described
    in `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The duration predictor predicts a duration of each frame in log domain
    from the hidden embeddings of encoder.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    Note:
        The calculation domain of outputs is different
        between in `forward` and in `inference`. In `forward`,
        the outputs are calculated in log domain but in `inference`,
        those are calculated in linear domain.

    """

    def __init__(self, input_dim, n_layers=2, n_chans=384, kernel_size=3, dropout_rate=0.1, offset=1.0):
        """
        Initialize duration predictor module.

        Args:
            input_dim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
            offset (float, optional): Offset value to avoid nan in log domain.

        """
        # self.experts = nn.ModuleDict()
        # for idx in range(self.num_experts):
        #     self.experts[f"expert_{idx}"] = expert_class(config, ffn_dim)
        super(DurationPredictor, self).__init__()
        self.offset = offset
        self.conv_layers = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = input_dim if idx == 0 else n_chans
            layer = torch.nn.ModuleList([
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate),
            ])
            self.conv_layers.append(layer)
        self.linear = torch.nn.Linear(n_chans, 1)

    def _forward(self, xs, x_masks=None, is_inference=False):
        # (batch_size, input_dim, max_text_length)
        xs = xs.transpose(1, -1)  
        for layer in self.conv_layers:
            for module in layer:
                xs = module(xs)

        # NOTE: calculate in log domain, (batch_size, max_text_length)
        xs = self.linear(xs.transpose(1, -1)).squeeze(-1)  

        if is_inference:
            # NOTE: calculate in linear domain
            xs = torch.clamp(torch.round(xs.exp() - self.offset), min=0).long()

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs

    def forward(self, xs, x_masks=None):
        """
        Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (batch_size, max_text_length, input_dim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (batch_size, max_text_length).

        Returns:
            Tensor: Batch of predicted durations in log domain (batch_size, max_text_length).

        """
        return self._forward(xs, x_masks, False)

    def inference(self, xs, x_masks=None):
        """
        Inference duration.

        Args:
            xs (Tensor): Batch of input sequences (batch_size, max_text_length, input_dim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (batch_size, max_text_length).

        Returns:
            LongTensor: Batch of predicted durations in linear domain (batch_size, max_text_length).

        """
        return self._forward(xs, x_masks, True)


class LengthRegulator(torch.nn.Module):
    """
    Length regulator module for feed-forward Transformer.

    This is a module of length regulator described in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.
    The length regulator expands char or
    phoneme-level embedding features to frame-level by repeating each
    feature based on the corresponding predicted durations.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf

    """

    def __init__(self, pad_value=0.0):
        """
        Initialize length regulator module.

        Args:
            pad_value (float, optional): Value used for padding.
        """
        super(LengthRegulator, self).__init__()
        self.pad_value = pad_value

    def forward(self, xs, target_durations, alpha=1.0):
        """
        Calculate forward propagation.

        Args:
            xs (Tensor): Batch of sequences of char or phoneme embeddings (batch_size, max_text_length, embedding_dim).
            target_durations (LongTensor): Batch of durations of each frame (batch_size, T).
            alpha (float, optional): Alpha value to control speed of speech.

        Returns:
            Tensor: replicated input tensor based on durations (batch_size, T*, embedding_dim).
        """

        if alpha != 1.0:
            assert alpha > 0
            target_durations = torch.round(target_durations.float() * alpha).long()

        if target_durations.sum() == 0:
            target_durations[target_durations.sum(dim=1).eq(0)] = 1

        return pad_list([self._repeat_one_sequence(x, d) for x, d in zip(xs, target_durations)], self.pad_value)

    def _repeat_one_sequence(self, x, d):
        """
        Repeat each frame according to duration
        """
        return torch.repeat_interleave(x, d, dim=0)


class PostNet(torch.nn.Module):
    """
    Postnet module for Spectrogram prediction network.

    This is a module of Postnet in Spectrogram prediction network,
    which described in `Natural TTS Synthesis by
    Conditioning WaveNet on Mel Spectrogram Predictions`_.
    The Postnet refines the predicted
    Mel-filterbank of the decoder,
    which helps to compensate the detail sturcture of spectrogram.

    .. _`Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions`:
       https://arxiv.org/abs/1712.05884
    """

    def __init__(self, output_dim, n_layers=5, n_chans=512, n_filts=5, dropout_rate=0.5, use_batch_norm=True):
        """
        Initialize postnet module.

        Args:
            output_dim (int): Dimension of the outputs.
            n_layers (int, optional): The number of layers.
            n_filts (int, optional): The number of filter size.
            n_units (int, optional): The number of filter channels.
            use_batch_norm (bool, optional): Whether to use batch normalization..
            dropout_rate (float, optional): Dropout rate..
        """
        super(PostNet, self).__init__()
        self.postnet = torch.nn.ModuleList()
        for n_layer in range(n_layers):
            is_last_layer = n_layer == n_layers - 1
            ichans = output_dim if n_layer == 0 else n_chans
            ochans = n_chans if n_layer < n_layers - 1 else output_dim
            
            layer = nn.ModuleList()
            layer.append(nn.Conv1d(ichans, ochans, n_filts, stride=1, padding=(n_filts - 1) // 2, bias=False))
            if use_batch_norm:
                num_groups = 32 if n_layer < n_layers - 1 else 20
                num_channels = output_dim if n_layer == n_layers - 1 else n_chans
                layer.append(nn.GroupNorm(num_groups=num_groups, num_channels=num_channels))
            if not is_last_layer:
                layer.append(nn.Tanh())
            layer.append(nn.Dropout(dropout_rate))

            self.postnet.append(layer)

    def forward(self, xs):
        """
        Calculate forward propagation.

        Args:
            xs (Tensor): Batch of the sequences of padded input tensors (batch_size, input_dim, max_text_length).

        Returns:
            Tensor: Batch of padded output tensor. (batch_size, output_dim, max_text_length).
        """
        for layer in self.postnet:
            for module in layer:
                xs = module(xs)
        return xs


class LayerNorm(torch.nn.LayerNorm):
    """Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.

    """

    def __init__(self, nout, dim=-1):
        """Construct an LayerNorm object."""
        super(LayerNorm, self).__init__(nout, eps=1e-12)
        self.dim = dim

    def forward(self, x):
        """Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.

        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return (
            super(LayerNorm, self)
            .forward(x.transpose(self.dim, -1))
            .transpose(self.dim, -1)
        )


class VariancePredictor(torch.nn.Module):
    """
    Variance predictor module.

    This is a module of variance predictor described in `FastSpeech 2:
    Fast and High-Quality End-to-End Text to Speech`_.

    .. _`FastSpeech 2: Fast and High-Quality End-to-End Text to Speech`:
        https://arxiv.org/abs/2006.04558

    """

    def __init__(
        self,
        input_dim,
        n_layers=2,
        n_chans=384,
        kernel_size=3,
        bias=True,
        dropout_rate=0.5,
    ):
        """
        Initilize duration predictor module.

        Args:
            input_dim (int): Input dimension.
            n_layers (int, optional): Number of convolutional layers.
            n_chans (int, optional): Number of channels of convolutional layers.
            kernel_size (int, optional): Kernel size of convolutional layers.
            dropout_rate (float, optional): Dropout rate.
        """
        super().__init__()
        self.conv_layers = torch.nn.ModuleList()
        for idx in range(n_layers):
            in_chans = input_dim if idx == 0 else n_chans
            layer = torch.nn.ModuleList([
                    torch.nn.Conv1d(
                        in_chans,
                        n_chans,
                        kernel_size,
                        stride=1,
                        padding=(kernel_size - 1) // 2,
                        bias=bias,
                    ),
                    torch.nn.ReLU(),
                    LayerNorm(n_chans, dim=1),
                    torch.nn.Dropout(dropout_rate)]
                )
            self.conv_layers.append(layer)
        self.linear = torch.nn.Linear(n_chans, 1)

    def forward(self, xs, x_masks=None):
        """
        Calculate forward propagation.

        Args:
            xs (Tensor): Batch of input sequences (batch_size, max_text_length, input_dim).
            x_masks (ByteTensor, optional):
                Batch of masks indicating padded part (batch_size, max_text_length).

        Returns:
            Tensor: Batch of predicted sequences (batch_size, max_text_length, 1).
        """
        # (batch_size, input_dim, max_text_length)
        xs = xs.transpose(1, -1)  
        for layer in self.conv_layers:
            for module in layer:
                xs = module(xs)  

        xs = self.linear(xs.transpose(1, 2)) 

        if x_masks is not None:
            xs = xs.masked_fill(x_masks, 0.0)

        return xs
    
class VarianceEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels=1,
        out_channels=384,
        kernel_size=1,
        padding=0,
        dropout_rate=0.0,
    ):
        super().__init__()
        self.conv = torch.nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
            )
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, xs):
        xs = xs.transpose(1, 2)
        xs = self.conv(xs)
        xs = self.dropout(xs)
        xs = xs.transpose(1, 2)
        return xs


class DurationPredictorLoss(torch.nn.Module):
    """
    Loss function module for duration predictor.

    The loss value is Calculated in log domain to make it Gaussian.

    """

    def __init__(self, offset=1.0, reduction="mean"):
        """
        Args:
            offset (float, optional): Offset value to avoid nan in log domain.
            reduction (str): Reduction type in loss calculation.

        """
        super(DurationPredictorLoss, self).__init__()
        self.criterion = torch.nn.MSELoss(reduction=reduction)
        self.offset = offset

    def forward(self, outputs, targets):
        """
        Calculate forward propagation.

        Args:
            outputs (Tensor): Batch of prediction durations in log domain (batch_size, T)
            targets (LongTensor): Batch of groundtruth durations in linear domain (batch_size, T)

        Returns:
            Tensor: Mean squared error loss value.

        Note:
            `outputs` is in log domain but `targets` is in linear domain.

        """
        # NOTE: outputs is in log domain while targets in linear
        targets = torch.log(targets.float() + self.offset)
        loss = self.criterion(outputs, targets)

        return loss


class FastSpeech2ConformerLoss(torch.nn.Module):
    def __init__(self, use_masking=True, use_weighted_masking=False):
        """
        use_masking (bool):
            Whether to apply masking for padded part in loss calculation.
        use_weighted_masking (bool):
            Whether to weighted masking in loss calculation.
        """
        super().__init__()

        assert (use_masking != use_weighted_masking) or not use_masking
        self.use_masking = use_masking
        self.use_weighted_masking = use_weighted_masking

        # define criterions
        reduction = "none" if self.use_weighted_masking else "mean"
        self.l1_criterion = torch.nn.L1Loss(reduction=reduction)
        self.mse_criterion = torch.nn.MSELoss(reduction=reduction)
        self.duration_criterion = DurationPredictorLoss(reduction=reduction)

    def forward(
        self,
        outputs_after_postnet,
        outputs_before_postnet,
        duration_outputs,
        pitch_outputs,
        energy_outputs,
        target_spectrograms,
        target_durations,
        target_pitch,
        target_energy,
        input_lengths,
        target_lengths,
    ):
        """
        Args:
            outputs_after_postnet (Tensor): Batch of outputs after postnets (batch_size, max_spectrogram_length, output_dim).
            outputs_before_postnet (Tensor): Batch of outputs before postnets (batch_size, max_spectrogram_length, output_dim).
            duration_outputs (LongTensor): Batch of outputs of duration predictor (batch_size, max_text_length).
            pitch_outputs (Tensor): Batch of outputs of pitch predictor (batch_size, max_text_length, 1).
            energy_outputs (Tensor): Batch of outputs of energy predictor (batch_size, max_text_length, 1).
            target_spectrograms (Tensor): Batch of target features (batch_size, max_spectrogram_length, output_dim).
            target_durations (LongTensor): Batch of durations (batch_size, max_text_length).
            target_pitch (Tensor): Batch of target token-averaged pitch (batch_size, max_text_length, 1).
            target_energy (Tensor): Batch of target token-averaged energy (batch_size, max_text_length, 1).
            input_lengths (LongTensor): Batch of the lengths of each input (batch_size,).
            target_lengths (LongTensor): Batch of the lengths of each target (batch_size,).

        Returns:
            Tensor: L1 loss value.
            Tensor: Duration predictor loss value.
            Tensor: Pitch predictor loss value.
            Tensor: Energy predictor loss value.

        """
        # apply mask to remove padded part
        if self.use_masking:
            out_masks = make_non_pad_mask(target_lengths).unsqueeze(-1).to(target_spectrograms.device)
            outputs_before_postnet = outputs_before_postnet.masked_select(out_masks)
            if outputs_after_postnet is not None:
                outputs_after_postnet = outputs_after_postnet.masked_select(out_masks)
            target_spectrograms = target_spectrograms.masked_select(out_masks)
            duration_masks = make_non_pad_mask(input_lengths).to(target_spectrograms.device)
            duration_outputs = duration_outputs.masked_select(duration_masks)
            target_durations = target_durations.masked_select(duration_masks)
            pitch_masks = make_non_pad_mask(input_lengths).unsqueeze(-1).to(target_spectrograms.device)
            pitch_outputs = pitch_outputs.masked_select(pitch_masks)
            energy_outputs = energy_outputs.masked_select(pitch_masks)
            target_pitch = target_pitch.masked_select(pitch_masks)
            target_energy = target_energy.masked_select(pitch_masks)

        # calculate loss
        l1_loss = self.l1_criterion(outputs_before_postnet, target_spectrograms)
        if outputs_after_postnet is not None:
            l1_loss = l1_loss + self.l1_criterion(outputs_after_postnet, target_spectrograms)
        duration_loss = self.duration_criterion(duration_outputs, target_durations)
        pitch_loss = self.mse_criterion(pitch_outputs, target_pitch)
        energy_loss = self.mse_criterion(energy_outputs, target_energy)

        # make weighted mask and apply it
        if self.use_weighted_masking:
            out_masks = make_non_pad_mask(target_lengths).unsqueeze(-1).to(target_spectrograms.device)
            out_masks = torch.nn.functional.pad(
                out_masks.transpose(1, 2), [0, target_spectrograms.size(1) - out_masks.size(1), 0, 0, 0, 0], value=False
            ).transpose(1, 2)

            out_weights = out_masks.float() / out_masks.sum(dim=1, keepdim=True).float()
            out_weights /= target_spectrograms.size(0) * target_spectrograms.size(2)
            duration_masks = make_non_pad_mask(input_lengths).to(target_spectrograms.device)
            duration_weights = duration_masks.float() / duration_masks.sum(dim=1, keepdim=True).float()
            duration_weights /= target_durations.size(0)

            # apply weight
            l1_loss = l1_loss.mul(out_weights).masked_select(out_masks).sum()
            duration_loss = duration_loss.mul(duration_weights).masked_select(duration_masks).sum()
            pitch_masks = duration_masks.unsqueeze(-1)
            pitch_weights = duration_weights.unsqueeze(-1)
            pitch_loss = pitch_loss.mul(pitch_weights).masked_select(pitch_masks).sum()
            energy_loss = energy_loss.mul(pitch_weights).masked_select(pitch_masks).sum()

        return l1_loss, duration_loss, pitch_loss, energy_loss


class FastSpeech2ConformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FastSpeech2ConformerConfig
    base_model_prefix = "fastspeech2_conformer"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        std = 1 # self.config.initializer_range
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            # if module.padding_idx is not None:
            #     module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.xavier_uniform_(module.weight, nn.init.calculate_gain("relu"))
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)


    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, FastSpeech2ConformerEncoder):
            module.gradient_checkpointing = value


class FastSpeech2ConformerModel(FastSpeech2ConformerPreTrainedModel):
    """
    FastSpeech 2 module.

    This is a module of FastSpeech 2 described in FastSpeech 2: Fast and
    High-Quality End-to-End Text to Speech. Instead of quantized pitch and
    energy, we use token-averaged value introduced in FastPitch: Parallel
    Text-to-speech with Pitch Prediction. The encoder and decoder are Conformers
    instead of regular Transformers.

        https://arxiv.org/abs/2006.04558
        https://arxiv.org/abs/2006.06873
        https://arxiv.org/pdf/2005.08100

    """

    def __init__(self, config):
        # config.utt_embed_dim=None,  # confirm this, previously was 64
        # config.lang_embs=None,      # confirm this, previously was 8000
        super().__init__(config)
        self.config = config

        # store hyperparameters
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        self.acoustic_dim = config.acoustic_dim
        self.eos = config.input_dim - 1  # replace with self.eos_token_id
        self.reduction_factor = config.reduction_factor
        self.stop_gradient_from_pitch_predictor = config.stop_gradient_from_pitch_predictor
        self.stop_gradient_from_energy_predictor = config.stop_gradient_from_energy_predictor
        self.multilingual_model = config.lang_embs is not None
        self.multispeaker_model = config.utt_embed_dim is not None



        # define encoder
        encoder_input_layer = torch.nn.Embedding(num_embeddings=self.input_dim, embedding_dim=self.acoustic_dim, padding_idx=0)

        self.encoder = FastSpeech2ConformerEncoder(
            attention_dim=self.acoustic_dim,
            attention_heads=config.num_attention_heads,
            linear_units=config.encoder_linear_units,
            num_blocks=config.encoder_layers,
            input_layer=encoder_input_layer,
            dropout_rate=config.transformer_enc_dropout_rate,
            positional_dropout_rate=config.transformer_enc_positional_dropout_rate,
            attention_dropout_rate=config.transformer_enc_attn_dropout_rate,
            normalize_before=config.encoder_normalize_before,
            concat_after=config.encoder_concat_after,
            positionwise_conv_kernel_size=config.positionwise_conv_kernel_size,
            macaron_style=config.use_macaron_style_in_conformer,
            use_cnn_module=config.use_cnn_in_conformer,
            cnn_module_kernel=config.conformer_enc_kernel_size,
            zero_triu=False,
            utt_embed=config.utt_embed_dim,
            lang_embs=config.lang_embs,
        )

        self.duration_predictor = DurationPredictor(
            input_dim=self.acoustic_dim,
            n_layers=config.duration_predictor_layers,
            n_chans=config.duration_predictor_chans,
            kernel_size=config.duration_predictor_kernel_size,
            dropout_rate=config.duration_predictor_dropout_rate,
        )

        self.pitch_predictor = VariancePredictor(
            input_dim=self.acoustic_dim,
            n_layers=config.pitch_predictor_layers,
            n_chans=config.pitch_predictor_chans,
            kernel_size=config.pitch_predictor_kernel_size,
            dropout_rate=config.pitch_predictor_dropout,
        )
        # continuous pitch + FastPitch style avg
        self.pitch_embed = VarianceEmbedding(
                out_channels=self.acoustic_dim,
                kernel_size=config.pitch_embed_kernel_size,
                padding=(config.pitch_embed_kernel_size - 1) // 2,
                dropout_rate=config.pitch_embed_dropout)

        self.energy_predictor = VariancePredictor(
            input_dim=self.acoustic_dim,
            n_layers=config.energy_predictor_layers,
            n_chans=config.energy_predictor_chans,
            kernel_size=config.energy_predictor_kernel_size,
            dropout_rate=config.energy_predictor_dropout,
        )
        # continuous energy + FastPitch style avg
        self.energy_embed = VarianceEmbedding(
                out_channels=self.acoustic_dim,
                kernel_size=config.energy_embed_kernel_size,
                padding=(config.energy_embed_kernel_size - 1) // 2,
                dropout_rate=config.energy_embed_dropout
                )

        self.length_regulator = LengthRegulator()

        self.decoder = FastSpeech2ConformerEncoder(
            attention_dim=self.acoustic_dim,
            attention_heads=config.num_attention_heads,
            linear_units=config.decoder_linear_units,
            num_blocks=config.decoder_layers,
            input_layer=None,
            dropout_rate=config.transformer_dec_dropout_rate,
            positional_dropout_rate=config.transformer_dec_positional_dropout_rate,
            attention_dropout_rate=config.transformer_dec_attn_dropout_rate,
            normalize_before=config.decoder_normalize_before,
            concat_after=config.decoder_concat_after,
            positionwise_conv_kernel_size=config.positionwise_conv_kernel_size,
            macaron_style=config.use_macaron_style_in_conformer,
            use_cnn_module=config.use_cnn_in_conformer,
            cnn_module_kernel=config.conformer_dec_kernel_size,
            utt_embed=config.utt_embed_dim,
            type="Decoder"
        )

        self.feat_out = torch.nn.Linear(self.acoustic_dim, self.output_dim * config.reduction_factor)

        self.postnet = PostNet(
            output_dim=self.output_dim,
            n_layers=config.postnet_layers,
            n_chans=config.postnet_chans,
            n_filts=config.postnet_filts,
            use_batch_norm=config.use_batch_norm,
            dropout_rate=config.postnet_dropout_rate,
        )

        self.criterion = FastSpeech2ConformerLoss(
            use_masking=config.use_masking, use_weighted_masking=config.use_weighted_masking
        )

        # self.post_init()

    def forward(
        self,
        input_ids,
        input_lengths,
        target_spectrograms,
        speech_lengths,
        target_durations,
        target_pitch,
        target_energy,
        utterance_embedding=None,
        return_mels=False,
        lang_ids=None,
    ):
        """
        Calculate forward propagation.

        Args:
            return_mels: whether to return the predicted spectrogram
            input_ids (LongTensor): Batch of padded text vectors (batch_size, max_text_length).
            input_lengths (LongTensor): Batch of lengths of each input (batch_size,).
            target_spectrograms (Tensor): Batch of padded target features (batch_size, max_spectrogram_length, output_dim).
            speech_lengths (LongTensor): Batch of the lengths of each target (batch_size,).
            target_durations (LongTensor): Batch of padded durations (batch_size, max_text_length + 1).
            target_pitch (Tensor): Batch of padded token-averaged pitch (batch_size, max_text_length + 1, 1).
            target_energy (Tensor): Batch of padded token-averaged energy (batch_size, max_text_length + 1, 1).

        Returns:
            Tensor: Loss scalar value.
            Dict: Statistics to be monitored.
            Tensor: Weight value.
        """
        # Texts include EOS token from the teacher model already in this version

        # forward propagation
        outputs = self._forward( # probably going to change to output_dict
            input_ids,
            input_lengths,
            speech_lengths,
            target_durations,
            target_pitch,
            target_energy,
            utterance_embedding=utterance_embedding,
            is_inference=False,
            lang_ids=lang_ids,
        )
        outputs_before_postnet, outputs_after_postnet, duration_outputs, pitch_outputs, energy_outputs = outputs
        # modify mod part of groundtruth (speaking pace)
        if self.reduction_factor > 1:
            speech_lengths = speech_lengths.new([olen - olen % self.reduction_factor for olen in speech_lengths])

        # calculate loss
        l1_loss, duration_loss, pitch_loss, energy_loss = self.criterion(
            outputs_after_postnet=outputs_after_postnet,
            outputs_before_postnet=outputs_before_postnet,
            duration_outputs=duration_outputs,
            pitch_outputs=pitch_outputs,
            energy_outputs=energy_outputs,
            target_spectrograms=target_spectrograms,
            target_durations=target_durations,
            target_pitch=target_pitch,
            target_energy=target_energy,
            input_lengths=input_lengths,
            target_lengths=speech_lengths,
        )
        loss = l1_loss + duration_loss + pitch_loss + energy_loss

        if return_mels:
            return loss, outputs_after_postnet
        return loss

    def _forward(
        self,
        input_ids,
        input_lengths,
        spectrogram_lengths=None,
        target_durations=None,
        target_pitch=None,
        target_energy=None,
        is_inference=False,
        alpha=1.0,
        utterance_embedding=None,
        lang_ids=None,
    ):
        if not self.multilingual_model:
            lang_ids = None

        if not self.multispeaker_model:
            utterance_embedding = None

        # forward encoder
        text_masks = self._source_mask(input_lengths)
        
        # (batch_size, max_text_length, acoustic_dim)
        encoded_texts, _ = self.encoder(
            input_ids, text_masks, utterance_embedding=utterance_embedding, lang_ids=lang_ids
        )

        # forward duration predictor and variance predictors
        d_masks = make_pad_mask(input_lengths, device=input_lengths.device)

        if self.stop_gradient_from_pitch_predictor:
            pitch_predictions = self.pitch_predictor(encoded_texts.detach(), d_masks.unsqueeze(-1))
        else:
            pitch_predictions = self.pitch_predictor(encoded_texts, d_masks.unsqueeze(-1))

        if self.stop_gradient_from_energy_predictor:
            energy_predictions = self.energy_predictor(encoded_texts.detach(), d_masks.unsqueeze(-1))
        else:
            energy_predictions = self.energy_predictor(encoded_texts, d_masks.unsqueeze(-1))

        if is_inference:
            duration_predictions = self.duration_predictor.inference(encoded_texts, d_masks)  # (batch_size, max_text_length)
            
            # use prediction in inference
            embedded_pitch_curve = self.pitch_embed(pitch_predictions)
            embedded_energy_curve = self.energy_embed(energy_predictions)
            encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve
            # (batch_size, max_spectrogram_length, acoustic_dim)
            encoded_texts = self.length_regulator(encoded_texts, duration_predictions, alpha)
        else:
            duration_predictions = self.duration_predictor(encoded_texts, d_masks)

            # use groundtruth in training
            embedded_pitch_curve = self.pitch_embed(target_pitch)
            embedded_energy_curve = self.energy_embed(target_energy)
            encoded_texts = encoded_texts + embedded_energy_curve + embedded_pitch_curve
            # (batch_size, max_spectrogram_length, acoustic_dim)
            encoded_texts = self.length_regulator(encoded_texts, target_durations)

        # forward decoder
        if spectrogram_lengths is not None and not is_inference:
            if self.reduction_factor > 1:
                target_lengths_in = spectrogram_lengths.new([olen // self.reduction_factor for olen in spectrogram_lengths])
            else:
                target_lengths_in = spectrogram_lengths
            h_masks = self._source_mask(target_lengths_in)
        else:
            h_masks = None
        # (batch_size, max_spectrogram_length, acoustic_dim)
        zs, _ = self.decoder(encoded_texts, h_masks, utterance_embedding)
        
        test = zs.detach().clone()
        torch.save(test, 'hf-decoder_output.pt')
        
        # (batch_size, max_spectrogram_length, output_dim)
        outputs_before_postnet = self.feat_out(zs).view(zs.size(0), -1, self.output_dim)
        
        test = outputs_before_postnet.detach().clone()
        torch.save(test, 'hf-outputs_before_postnet.pt')

        # postnet -> (batch_size, max_spectrogram_length//r * r, output_dim)
        outputs_after_postnet = outputs_before_postnet + self.postnet(outputs_before_postnet.transpose(1, 2)).transpose(1, 2)
        
        test = outputs_after_postnet.detach().clone()
        torch.save(test, 'hf-outputs_after_postnet.pt')
        
        test = duration_predictions.detach().clone()
        torch.save(test, 'hf-duration_predictions.pt')
        
        test = pitch_predictions.detach().clone()
        torch.save(test, 'hf-pitch_predictions.pt')
        
        test = energy_predictions.detach().clone()
        torch.save(test, 'hf-energy_predictions.pt')

        return outputs_before_postnet, outputs_after_postnet, duration_predictions, pitch_predictions, energy_predictions

    # This is now .generate(), also add .generate_speech()
    def inference(
        self,
        text,
        speech=None,
        durations=None,
        pitch=None,
        energy=None,
        alpha=1.0,
        utterance_embedding=None,
        return_duration_pitch_energy=False,
        lang_id=None,
    ):
        """
        Generate the sequence of features given the sequences of characters.

        Args:
            text (LongTensor): Input sequence of characters (text_length,).
            speech (Tensor, optional): Feature sequence to extract style (batch_size, input_dim).
            durations (LongTensor, optional): Groundtruth of duration (text_length + 1,).
            pitch (Tensor, optional): Groundtruth of token-averaged pitch (text_length + 1, 1).
            energy (Tensor, optional): Groundtruth of token-averaged energy (text_length + 1, 1).
            alpha (float, optional): Alpha to control the speed.
            use_teacher_forcing (bool, optional): Whether to use teacher forcing.
                If true, groundtruth of duration, pitch and energy will be used.
            return_duration_pitch_energy: whether to return the list of predicted durations for nicer plotting

        Returns:
            Tensor: Output sequence of features (output_spectrogram_length, output_dim). # return_dict

        """
        self.eval()
        x, y = text, speech
        d, p, e = durations, pitch, energy
        
        # add eos at the last of sequence
        x = torch.nn.functional.pad(x, [0, 1], "constant", self.eos)

        # setup batch axis
        input_lengths = torch.tensor([x.shape[0]], dtype=torch.long, device=x.device)
        xs, ys = x.unsqueeze(0), None
        
        if y is not None:
            ys = y.unsqueeze(0)
        if lang_id is not None:
            lang_id = lang_id.unsqueeze(0)
        if utterance_embedding is not None:
            utterance_embedding.unsqueeze(0)
        
        # (1, L, output_dim)
        outputs_before_postnet, outputs_after_postnet, duration_outputs, pitch_predictions, energy_predictions = self._forward(
            xs,
            input_lengths,
            ys,
            is_inference=True,
            alpha=alpha,
            utterance_embedding=utterance_embedding,
            lang_ids=lang_id,
        )

        self.train()
        if return_duration_pitch_energy:
            return outputs_after_postnet[0], duration_outputs[0], pitch_predictions[0], energy_predictions[0]
        return outputs_after_postnet[0]

    def _source_mask(self, input_lengths):
        """
        Make masks for self-attention.

        Args:
            input_lengths (LongTensor): Batch of lengths (batch_size,).

        Returns:
            Tensor: Mask tensor for self-attention.

        """
        x_masks = make_non_pad_mask(input_lengths, device=input_lengths.device)
        return x_masks.unsqueeze(-2)

    @torch.no_grad()
    def generate_speech(
        self,
        feat_gen,
        vocoder: Optional[nn.Module] = None,
    ):
        output_dict = {}
        # apply vocoder (mel-to-wav)
        if vocoder is not None:
            # confirm this from ims toucan, previously in espnet was input_feat = output_dict["feat_gen"]
            wav = vocoder(feat_gen)
            output_dict.update(wav=wav)

        return output_dict


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention layer.

    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
    """

    def __init__(self, n_head, n_feat, dropout_rate):
        """
        Construct an MultiHeadedAttention object.
        """
        super(MultiHeadedAttention, self).__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_feat)
        self.linear_k = nn.Linear(n_feat, n_feat)
        self.linear_v = nn.Linear(n_feat, n_feat)
        self.linear_out = nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward_qkv(self, query, key, value):
        """
        Transform query, key and value.

        Args:
            query (torch.Tensor): Query tensor (batch, time1, size).
            key (torch.Tensor): Key tensor (batch, time2, size).
            value (torch.Tensor): Value tensor (batch, time2, size).

        Returns:
            torch.Tensor: Transformed query tensor (batch, n_head, time1, d_k).
            torch.Tensor: Transformed key tensor (batch, n_head, time2, d_k).
            torch.Tensor: Transformed value tensor (batch, n_head, time2, d_k).
        """
        n_batch = query.size(0)
        q = self.linear_q(query).view(n_batch, -1, self.h, self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h, self.d_k)
        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)
        # (batch_size, head, time1, d_k)
        q = q.transpose(1, 2)
        # (batch_size, head, time2, d_k)  
        k = k.transpose(1, 2)
        # (batch_size, head, time2, d_k)
        v = v.transpose(1, 2)

        return q, k, v

    def forward_attention(self, value, scores, mask):
        """
        Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (batch, 1, time2) or (batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (batch, time1, d_model)
                weighted by the attention score (batch, time1, time2).
        """
        n_batch = value.size(0)
        if mask is not None:
            # (batch_size, 1, *, time2)
            mask = mask.unsqueeze(1).eq(0)
            min_value = float(numpy.finfo(torch.tensor(0, dtype=scores.dtype).numpy().dtype).min)
            scores = scores.masked_fill(mask, min_value)
            # (batch_size, head, time1, time2)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(mask, 0.0)
        else:
            # (batch_size, head, time1, time2)
            self.attn = torch.softmax(scores, dim=-1)

        p_attn = self.dropout(self.attn)
        # (batch_size, head, time1, d_k)
        x = torch.matmul(p_attn, value)
        # (batch_size, time1, d_model)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)

        # (batch_size, time1, d_model)
        return self.linear_out(x)  

    def forward(self, query, key, value, mask):
        """
        Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (batch, time1, size).
            key (torch.Tensor): Key tensor (batch, time2, size).
            value (torch.Tensor): Value tensor (batch, time2, size).
            mask (torch.Tensor): Mask tensor (batch, 1, time2) or
                (batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask)


class RelPositionMultiHeadedAttention(MultiHeadedAttention):
    """
    Multi-Head Attention layer with relative position encoding.
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    Paper: https://arxiv.org/abs/1901.02860
    Args:
        n_head (int): The number of heads.
        n_feat (int): The number of features.
        dropout_rate (float): Dropout rate.
        zero_triu (bool): Whether to zero the upper triangular part of attention matrix.
    """

    def __init__(self, n_head, n_feat, dropout_rate, zero_triu=False):
        """Construct an RelPositionMultiHeadedAttention object."""
        super().__init__(n_head, n_feat, dropout_rate)
        self.zero_triu = zero_triu
        # linear transformation for positional encoding
        self.linear_pos = nn.Linear(n_feat, n_feat, bias=False)
        # these two learnable bias are used in matrix c and matrix d
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        self.pos_bias_u = nn.Parameter(torch.Tensor(self.h, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.Tensor(self.h, self.d_k))
        torch.nn.init.xavier_uniform_(self.pos_bias_u)
        torch.nn.init.xavier_uniform_(self.pos_bias_v)

    def rel_shift(self, x):
        """
        Compute relative positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch_size, head, time1, 2*time1-1).
            time1 means the length of query vector.
        Returns:
            torch.Tensor: Output tensor.
        """
        zero_pad = torch.zeros((*x.size()[:3], 1), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=-1)

        x_padded = x_padded.view(*x.size()[:2], x.size(3) + 1, x.size(2))
        # only keep the positions from 0 to time2
        x = x_padded[:, :, 1:].view_as(x)[:, :, :, : x.size(-1) // 2 + 1]

        if self.zero_triu:
            ones = torch.ones((x.size(2), x.size(3)), device=x.device)
            x = x * torch.tril(ones, x.size(3) - x.size(2))[None, None, :, :]

        return x

    def forward(self, query, key, value, pos_emb, mask):
        """
        Compute 'Scaled Dot Product Attention' with rel. positional encoding.
        Args:
            query (torch.Tensor): Query tensor (batch, time1, size).
            key (torch.Tensor): Key tensor (batch, time2, size).
            value (torch.Tensor): Value tensor (batch, time2, size).
            pos_emb (torch.Tensor): Positional embedding tensor
                (batch, 2*time1-1, size).
            mask (torch.Tensor): Mask tensor (batch, 1, time2) or
                (batch, time1, time2).
        Returns:
            torch.Tensor: Output tensor (batch, time1, d_model).
        """
        q, k, v = self.forward_qkv(query, key, value)
        # (batch_size, time1, head, d_k)
        q = q.transpose(1, 2)

        n_batch_pos = pos_emb.size(0)
        p = self.linear_pos(pos_emb).view(n_batch_pos, -1, self.h, self.d_k)
        # (batch_size, head, 2*time1-1, d_k)
        p = p.transpose(1, 2)

        # (batch_size, head, time1, d_k)
        q_with_bias_u = (q + self.pos_bias_u).transpose(1, 2)
        # (batch_size, head, time1, d_k)
        q_with_bias_v = (q + self.pos_bias_v).transpose(1, 2)

        # compute attention score
        # first compute matrix a and matrix c
        # as described in https://arxiv.org/abs/1901.02860 Section 3.3
        # (batch_size, head, time1, time2)
        matrix_ac = torch.matmul(q_with_bias_u, k.transpose(-2, -1))

        # compute matrix b and matrix d
        # (batch_size, head, time1, 2*time1-1)
        matrix_bd = torch.matmul(q_with_bias_v, p.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)

        # (batch_size, head, time1, time2)
        scores = (matrix_ac + matrix_bd) / math.sqrt(self.d_k)

        return self.forward_attention(v, scores, mask)

class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x):
        """Return Swich activation function."""
        return x * torch.sigmoid(x)
    
class FastSpeech2ConformerConvolutionModule(nn.Module):
    """
    FastSpeech2ConformerEncoderConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.

    """

    def __init__(self, channels, kernel_size, bias=True):
        super(FastSpeech2ConformerConvolutionModule, self).__init__()
        # kernel_size should be an odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = Swish()

    def forward(self, x):
        """
        Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (batch, time, channels).

        Returns:
            torch.Tensor: Output tensor (batch, time, channels).

        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)

        # GLU mechanism, (batch_size, 2*channel, dim)
        x = self.pointwise_conv1(x)
        # (batch_size, channel, dim)
        x = nn.functional.glu(x, dim=1)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x)

        return x.transpose(1, 2)


class EncoderLayer(nn.Module):
    """
    Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
    """

    def __init__(
        self,
        size,
        self_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        encoder_layer = 1,
        type_ = "Encoder"
    ):
        super(EncoderLayer, self).__init__()
        self.encoder_layer = encoder_layer
        self.type_ = type_
        
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        
        # for the FNN module
        self.norm_ff = LayerNorm(size)  
        
        # for the MHA module
        self.norm_mha = LayerNorm(size)
        
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = LayerNorm(size)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            # for the CNN module
            self.norm_conv = LayerNorm(size)  
            
            # for the final output of the block
            self.norm_final = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.dropout_rate = dropout_rate
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)

    def forward(self, x_input, mask, cache=None):
        """
        Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (batch, time).
            cache (torch.Tensor): Cache tensor of the input (batch, time - 1, size).

        Returns:
            torch.Tensor: Output tensor (batch, time, size).
            torch.Tensor: Mask tensor (batch, time).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))

            if self.encoder_layer == 1 and self.type_ == "Encoder":
                test = x.detach().clone()
                torch.save(test, 'hf-macaron_layer_1.pt')

            if not self.normalize_before:
                x = self.norm_ff_macaron(x)
        
        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att = self.self_attn(x_q, x, x, pos_emb, mask)
        else:
            x_att = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x = residual + self.dropout(self.conv_module(x))
            if not self.normalize_before:
                x = self.norm_conv(x)
                
        if self.type_ == "Encoder":
            test = x.clone().detach() # tuple(x.clone().detach() for tensor in x_input)
            torch.save(test, f'hf-encoder_layer_{self.encoder_layer}_output.pt')

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb), mask

        return x, mask


class LayerNorm(torch.nn.LayerNorm):
    """
    Layer normalization module.

    Args:
        nout (int): Output dim size.
        dim (int): Dimension to be normalized.
    """

    def __init__(self, nout, dim=-1, eps=1e-12):
        """
        Construct an LayerNorm object.
        """
        super(LayerNorm, self).__init__(nout, eps=eps)
        self.dim = dim

    def forward(self, x):
        """
        Apply layer normalization.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor.
        """
        if self.dim == -1:
            return super(LayerNorm, self).forward(x)
        return super(LayerNorm, self).forward(x.transpose(1, -1)).transpose(1, -1)


class MultiLayeredConv1d(torch.nn.Module):
    """
    Multi-layered conv1d for Transformer block.

    This is a module of multi-layered conv1d designed
    to replace positionwise feed-forward network
    in Transformer block, which is introduced in
    `FastSpeech: Fast, Robust and Controllable Text to Speech`_.

    .. _`FastSpeech: Fast, Robust and Controllable Text to Speech`:
        https://arxiv.org/pdf/1905.09263.pdf
    """

    def __init__(self, in_chans, hidden_chans, kernel_size, dropout_rate, type_="Encoder", encoder_layer=0):
        """
        Initialize MultiLayeredConv1d module.

        Args:
            in_chans (int): Number of input channels.
            hidden_chans (int): Number of hidden channels.
            kernel_size (int): Kernel size of conv1d.
            dropout_rate (float): Dropout rate.
        """
        super(MultiLayeredConv1d, self).__init__()
        self.w_1 = torch.nn.Conv1d(
            in_chans,
            hidden_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.w_2 = torch.nn.Conv1d(
            hidden_chans,
            in_chans,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.type_ = type_
        self.encoder_layer = encoder_layer

    def forward(self, x):
        """
        Calculate forward propagation.

        Args:
            x (torch.Tensor): Batch of input tensors (batch_size, T, in_chans).

        Returns:
            torch.Tensor: Batch of output tensors (batch_size, T, hidden_chans).
        """
        x = torch.relu(self.w_1(x.transpose(-1, 1))).transpose(-1, 1)
        return self.w_2(x.transpose(-1, 1)).transpose(-1, 1)


class RelPositionalEncoding(torch.nn.Module):
    """
    Relative positional encoding module (new implementation).
    Details can be found in https://github.com/espnet/espnet/pull/2816.
    See : Appendix Batch in https://arxiv.org/abs/1901.02860
    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.
    """

    def __init__(self, d_model, dropout_rate, max_len=5000, type_="Encoder"):
        """
        Construct an PositionalEncoding object.
        """
        super(RelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))
        self.type_ = type_
        

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x):
        """
        Add positional encoding.
        Args:
            x (torch.Tensor): Input tensor (batch_size, time, `*`).
        Returns:
            torch.Tensor: Encoded tensor (batch_size, time, `*`).
        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.pe[
            :,
            self.pe.size(1) // 2 - x.size(1) + 1 : self.pe.size(1) // 2 + x.size(1),
        ]
        return self.dropout(x), self.dropout(pos_emb)


class FastSpeech2ConformerEncoder(torch.nn.Module):
    """
    FastSpeech2ConformerEncoder encoder module.

    Args:
        input_dim (int): Input dimension.
        attention_dim (int): Dimension of attention.
        attention_heads (int): The number of heads of multi head attention.
        linear_units (int): The number of units of position-wise feed forward.
        num_blocks (int): The number of decoder blocks.
        dropout_rate (float): Dropout rate.
        positional_dropout_rate (float): Dropout rate after adding positional encoding.
        attention_dropout_rate (float): Dropout rate in attention.
        input_layer (Union[str, torch.nn.Module]): Input layer type.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        positionwise_layer_type (str): "linear", "conv1d", or "conv1d-linear".
        positionwise_conv_kernel_size (int): Kernel size of positionwise conv1d layer.
        macaron_style (bool): Whether to use macaron style for positionwise layer.
        selfattention_layer_type (str): Conformer attention layer type.
        activation_type (str): Conformer activation function type.
        use_cnn_module (bool): Whether to use convolution module.
        cnn_module_kernel (int): Kernel size of convolution module.

    """

    def __init__(
        self,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        attention_dropout_rate=0.0,
        input_layer=None,
        normalize_before=True,
        concat_after=False,
        positionwise_conv_kernel_size=1,
        macaron_style=False,
        use_cnn_module=False,
        cnn_module_kernel=31,
        zero_triu=False,
        utt_embed=None,
        lang_embs=None,
        type="Encoder"
    ):
        super(FastSpeech2ConformerEncoder, self).__init__()
        self.type_ = type
        
        self.embed = input_layer
        self.pos_enc = RelPositionalEncoding(attention_dim, positional_dropout_rate, type_=self.type_)

        self.utt_embed = utt_embed
        if utt_embed is not None:
            self.hs_emb_projection = torch.nn.Linear(attention_dim + utt_embed, attention_dim)
        if lang_embs is not None:
            self.language_embedding = torch.nn.Embedding(num_embeddings=lang_embs, embedding_dim=attention_dim)

        # self-attention module definition
        encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, attention_dim, attention_dropout_rate, zero_triu)

        # feed-forward module definition
        positionwise_layer = MultiLayeredConv1d
        positionwise_layer_args = (
            attention_dim,
            linear_units,
            positionwise_conv_kernel_size,
            dropout_rate,
            self.type_
        )

        # convolution module definition
        convolution_layer = FastSpeech2ConformerConvolutionModule
        convolution_layer_args = (attention_dim, cnn_module_kernel)

        self.encoders = nn.ModuleList(
            [
                EncoderLayer(
                    attention_dim,
                    encoder_selfattn_layer(*encoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args, _ + 1) if macaron_style else None,
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                    encoder_layer=_ + 1,
                    type_=self.type_
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, xs, masks, utterance_embedding=None, lang_ids=None):
        """
        Encode input sequence.
        Args:
            xs (torch.Tensor): Input tensor (batch, time, input_dim).
            masks (torch.Tensor): Mask tensor (batch, time).
            utterance_embedding: embedding containing lots of conditioning signals
            lang_ids: ids of the languages per sample in the batch
        Returns:
            torch.Tensor: Output tensor (batch, time, attention_dim).
            torch.Tensor: Mask tensor (batch, time).
        """
        if self.embed is not None:
            xs = self.embed(xs)
        if lang_ids is not None:
            lang_embs = self.language_embedding(lang_ids)
            # offset phoneme representation by language specific offset
            xs = xs + lang_embs  

        xs = self.pos_enc(xs)
        
        if self.type_ == "Encoder":
            test = tuple(tensor.clone().detach() for tensor in xs)
            torch.save(test, 'hf-pre_encoders.pt')

        for encoder_layer in self.encoders:
            xs, masks = encoder_layer(xs, masks)
        
        if self.type_ == "Encoder":
            test = tuple(tensor.clone().detach() for tensor in xs)
            torch.save(test, 'hf-encoders_output.pt')

        if isinstance(xs, tuple):
            xs = xs[0]

        if self.utt_embed:
            xs = self._integrate_with_utt_embed(hs=xs, utt_embeddings=utterance_embedding)

        # xs = self.output_norm(xs) # confirm this, should it be deleted? for some reason the pretrained checkpoint didn't have this value
        return xs, masks

    def _integrate_with_utt_embed(self, hs, utt_embeddings):
        # concat hidden states with spk embeds and then apply projection
        embeddings_expanded = torch.nn.functional.normalize(utt_embeddings).unsqueeze(1).expand(-1, hs.size(1), -1)
        hs = self.hs_emb_projection(torch.cat([hs, embeddings_expanded], dim=-1))
        return hs
