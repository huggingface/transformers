# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import EMAConfig
from fairseq.models import FairseqEncoder, FairseqEncoderModel, register_model
from fairseq.models.ema import EMA
from fairseq.models.roberta.model import RobertaClassificationHead, RobertaLMHead
from fairseq.models.transformer import TransformerConfig, TransformerEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params

from omegaconf import II


logger = logging.getLogger(__name__)


@dataclass
class Data2VecTextConfig(FairseqDataclass):
    max_positions: int = II("task.tokens_per_sample")

    head_layers: int = 1

    transformer: TransformerConfig = TransformerConfig()

    load_checkpoint_heads: bool = field(
        default=False,
        metadata={"help": "(re-)register and load heads when loading checkpoints"},
    )

    loss_beta: float = field(default=0, metadata={"help": "beta for smooth l1 loss. 0 means use l2 loss"})
    loss_scale: Optional[float] = field(
        default=None,
        metadata={"help": "scale the reconstruction loss by this constant. if None then scales by 1/sqrt(dim)"},
    )
    average_top_k_layers: int = field(default=8, metadata={"help": "how many layers to average"})

    layer_norm_target_layer: bool = False
    instance_norm_target_layer: bool = False
    batch_norm_target_layer: bool = False
    instance_norm_targets: bool = False
    layer_norm_targets: bool = False

    ema_decay: float = field(default=0.999, metadata={"help": "initial ema decay rate"})
    ema_end_decay: float = field(default=0.9999, metadata={"help": "final ema decay rate"})

    # when to finish annealing ema decay rate
    ema_anneal_end_step: int = II("optimization.max_update")

    ema_transformer_layers_only: bool = field(
        default=True,
        metadata={"help": "whether to momentum update only the transformer layers"},
    )


def get_annealed_rate(start, end, curr_step, total_steps):
    r = end - start
    pct_remaining = 1 - curr_step / total_steps
    return end - r * pct_remaining


@register_model("data2vec_text", dataclass=Data2VecTextConfig)
class Data2VecTextModel(FairseqEncoderModel):
    def __init__(self, cfg: Data2VecTextConfig, encoder):
        super().__init__(encoder)
        self.cfg = cfg

        # We follow BERT's random weight initialization
        self.apply(init_bert_params)

        self.classification_heads = nn.ModuleDict()

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        encoder = Data2VecTextEncoder(cfg, task.source_dictionary, task.cfg.data)

        return cls(cfg, encoder)

    def forward(
        self,
        src_tokens,
        target_tokens=None,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):
        if classification_head_name is not None:
            features_only = True

        res = self.encoder(src_tokens, target_tokens, features_only, return_all_hiddens, **kwargs)

        if isinstance(res, tuple):
            x, extra = res
        else:
            return res

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, extra

    def get_normalized_probs(self, net_output, log_probs, sample=None):
        """Get normalized probabilities (or log probs) from a net's output."""
        logits = net_output[0].float()
        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                logger.warning(
                    're-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )
        self.classification_heads[name] = RobertaClassificationHead(
            input_dim=self.cfg.transformer.encoder.embed_dim,
            inner_dim=inner_dim or self.cfg.transformer.encoder.embed_dim,
            num_classes=num_classes,
            activation_fn="tanh",
            pooler_dropout=0,
        )

    @property
    def supported_targets(self):
        return {"self"}

    def upgrade_state_dict_named(self, state_dict, name):
        prefix = name + "." if name != "" else ""

        # rename decoder -> encoder before upgrading children modules
        for k in list(state_dict.keys()):
            if k.startswith(prefix + "decoder"):
                new_k = prefix + "encoder" + k[len(prefix + "decoder") :]
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

            if self.encoder.regression_head is not None:
                if ".lm_head." in k:
                    new_k = k.replace(".lm_head.", ".regression_head.")
                    state_dict[new_k] = state_dict[k]
                    del state_dict[k]
            else:
                if ".regression_head." in k:
                    del state_dict[k]

        # upgrade children modules
        super().upgrade_state_dict_named(state_dict, name)

        # Handle new classification heads present in the state dict.
        current_head_names = (
            []
            if not hasattr(self, "classification_heads") or self.classification_heads is None
            else self.classification_heads.keys()
        )
        keys_to_delete = []
        for k in state_dict.keys():
            if not k.startswith(prefix + "classification_heads."):
                continue

            head_name = k[len(prefix + "classification_heads.") :].split(".")[0]
            num_classes = state_dict[prefix + "classification_heads." + head_name + ".out_proj.weight"].size(0)
            inner_dim = state_dict[prefix + "classification_heads." + head_name + ".dense.weight"].size(0)

            if self.cfg.load_checkpoint_heads:
                if head_name not in current_head_names:
                    self.register_classification_head(head_name, num_classes, inner_dim)
            else:
                if head_name not in current_head_names:
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "not present in current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
                elif (
                    num_classes != self.classification_heads[head_name].out_proj.out_features
                    or inner_dim != self.classification_heads[head_name].dense.out_features
                ):
                    logger.warning(
                        "deleting classification head ({}) from checkpoint "
                        "with different dimensions than current model: {}".format(head_name, k)
                    )
                    keys_to_delete.append(k)
        for k in keys_to_delete:
            del state_dict[k]

        # Copy any newly-added classification heads into the state dict
        # with their current weights.
        if (
            hasattr(self, "classification_heads")
            and self.classification_heads is not None
            and len(self.classification_heads) > 0
        ):
            cur_state = self.classification_heads.state_dict()
            for k, v in cur_state.items():
                if prefix + "classification_heads." + k not in state_dict:
                    logger.info("Overwriting " + prefix + "classification_heads." + k)
                    state_dict[prefix + "classification_heads." + k] = v

            for k in list(state_dict.keys()):
                if k.startswith(prefix + "encoder.lm_head.") or k.startswith(prefix + "encoder.emb_head."):
                    del state_dict[k]

            self.encoder.lm_head = None

        if self.encoder.target_model is None:
            for k in list(state_dict.keys()):
                if k.startswith(prefix + "encoder.target_model."):
                    del state_dict[k]

        if (self.encoder.ema is None) and (prefix + "encoder._ema" in state_dict):
            del state_dict[prefix + "encoder._ema"]

    def remove_pretraining_modules(self, last_layer=None):
        self.encoder.lm_head = None
        self.encoder.regression_head = None
        self.encoder.ema = None
        self.classification_heads = None

        if last_layer is not None:
            self.encoder.sentence_encoder.layers = nn.ModuleList(
                l for i, l in enumerate(self.encoder.sentence_encoder.layers) if i <= last_layer
            )
            self.encoder.sentence_encoder.layer_norm = None


class Data2VecTextEncoder(FairseqEncoder):
    def __init__(self, cfg: Data2VecTextConfig, dictionary, task_data):
        super().__init__(dictionary)

        self.cfg = cfg

        embed_tokens = self.build_embedding(len(dictionary), cfg.transformer.encoder.embed_dim, dictionary.pad())

        self.sentence_encoder = self.build_encoder(cfg, dictionary, embed_tokens)
        self.mask_idx = dictionary.index("<mask>")
        assert self.mask_idx != dictionary.unk(), dictionary.symbols

        self.ema = None
        self.average_top_k_layers = cfg.average_top_k_layers
        self.loss_scale = cfg.loss_scale

        assert self.cfg.head_layers >= 1

        embed_dim = cfg.transformer.encoder.embed_dim
        curr_dim = embed_dim
        projs = []
        for i in range(self.cfg.head_layers - 1):
            next_dim = embed_dim * 2 if i == 0 else curr_dim
            projs.append(nn.Linear(curr_dim, next_dim))
            projs.append(nn.GELU())
            curr_dim = next_dim

        projs.append(nn.Linear(curr_dim, embed_dim))
        self.regression_head = nn.Sequential(*projs)

        self.num_updates = 0

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def build_encoder(self, cfg, dictionary, embed_tokens):
        encoder = TransformerEncoder(cfg.transformer, dictionary, embed_tokens, return_fc=True)
        encoder.apply(init_bert_params)
        return encoder

    def build_lm_head(self, embed_dim, output_dim, activation_fn, weight):
        return RobertaLMHead(embed_dim, output_dim, activation_fn, weight)

    def make_ema_teacher(self):
        ema_config = EMAConfig(
            ema_decay=self.cfg.ema_decay,
            ema_fp32=True,
        )
        skip_keys = set()
        if self.cfg.ema_transformer_layers_only:
            for k, _ in self.sentence_encoder.embed_positions.named_parameters():
                skip_keys.add(f"embed_tokens.{k}")
            for k, _ in self.sentence_encoder.embed_positions.named_parameters():
                skip_keys.add(f"embed_positions.{k}")
            if self.sentence_encoder.layernorm_embedding is not None:
                for (
                    k,
                    _,
                ) in self.sentence_encoder.layernorm_embedding.named_parameters():
                    skip_keys.add(f"layernorm_embedding.{k}")
            if self.sentence_encoder.layer_norm is not None:
                for k, _ in self.sentence_encoder.layer_norm.named_parameters():
                    skip_keys.add(f"layernorm_embedding.{k}")

        self.ema = EMA(
            self.sentence_encoder,
            ema_config,
            skip_keys=skip_keys,
        )

    def set_num_updates(self, num_updates):
        super().set_num_updates(num_updates)

        if self.ema is None and self.regression_head is not None:
            logger.info("making ema teacher")
            self.make_ema_teacher()
        elif self.training and self.ema is not None:
            if self.cfg.ema_decay != self.cfg.ema_end_decay:
                if num_updates >= self.cfg.ema_anneal_end_step:
                    decay = self.cfg.ema_end_decay
                else:
                    decay = get_annealed_rate(
                        self.cfg.ema_decay,
                        self.cfg.ema_end_decay,
                        num_updates,
                        self.cfg.ema_anneal_end_step,
                    )
                self.ema._set_decay(decay)
            if self.ema.get_decay() < 1:
                self.ema.step(self.sentence_encoder)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(destination, prefix, keep_vars)
        if self.ema is not None:
            state[prefix + "_ema"] = self.ema.fp32_params
        return state

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        if self.ema is not None:
            k = prefix + "_ema"
            assert k in state_dict
            self.ema.restore(state_dict[k], True)
            del state_dict[k]
        return super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)

    def forward(
        self,
        src_tokens,
        target_tokens=None,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states' is a list of hidden states. Note that the
                  hidden states have shape `(src_len, batch, vocab)`.
        """

        x, extra = self.extract_features(src_tokens, return_all_hiddens=return_all_hiddens)

        if features_only:
            return x, extra

        assert target_tokens is not None

        with torch.no_grad():
            # use EMA parameter as the teacher
            self.ema.model.eval()

            encoder_out = self.ema.model(
                target_tokens,
                return_all_hiddens=True,
            )
            y = encoder_out["fc_results"]

            y = y[-self.average_top_k_layers :]

            permuted = False
            if self.cfg.instance_norm_target_layer or self.cfg.batch_norm_target_layer:
                y = [tl.permute(1, 2, 0) for tl in y]  # TBC -> BCT
                permuted = True

            if self.cfg.batch_norm_target_layer:
                y = [F.batch_norm(tl.float(), running_mean=None, running_var=None, training=True) for tl in y]

            if self.cfg.instance_norm_target_layer:
                y = [F.instance_norm(tl.float()) for tl in y]

            if permuted:
                y = [tl.transpose(1, 2) for tl in y]  # BCT -> BTC

            if self.cfg.layer_norm_target_layer:
                y = [F.layer_norm(tl.float(), tl.shape[-1:]) for tl in y]

            y = sum(y) / len(y)

            if not permuted:
                y = y.transpose(0, 1)

            if self.cfg.layer_norm_targets:
                y = F.layer_norm(y.float(), y.shape[-1:])

            if self.cfg.instance_norm_targets:
                y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        masked_indices = src_tokens.eq(self.mask_idx)

        x = x[masked_indices]
        y = y[masked_indices]

        x = self.regression_head(x)

        sz = x.size(-1)
        if self.cfg.loss_beta == 0:
            loss = F.mse_loss(x.float(), y.float(), reduction="none").sum(dim=-1)
        else:
            loss = F.smooth_l1_loss(x.float(), y.float(), reduction="none", beta=self.cfg.loss_beta).sum(dim=-1)

        result = {
            "losses": {
                "main": loss.sum() / math.sqrt(sz) if self.loss_scale <= 0 else loss.sum() * self.loss_scale,
            },
            "sample_size": loss.numel(),
        }

        # logging other values
        other_logs = {"ema_decay": self.ema.get_decay() * 1000}
        result["logs"] = other_logs
        return result

    def extract_features(self, src_tokens, return_all_hiddens=False, **kwargs):
        encoder_out = self.sentence_encoder(
            src_tokens,
            return_all_hiddens=return_all_hiddens,
            token_embeddings=kwargs.get("token_embeddings", None),
        )
        # T x B x C -> B x T x C
        features = encoder_out["encoder_out"][0].transpose(0, 1)
        inner_states = encoder_out["encoder_states"] if return_all_hiddens else None
        return features, {
            "inner_states": inner_states,
            "encoder_embedding": encoder_out["encoder_embedding"][0],
        }

    def output_layer(self, features, masked_tokens=None, **unused):
        return self.lm_head(features, masked_tokens)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.cfg.max_positions
