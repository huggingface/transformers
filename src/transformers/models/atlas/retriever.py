# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch

from ..bert.modeling_bert import BertModel

EMBEDDINGS_DIM: int = 768


class Contriever(BertModel):
    def __init__(self, config, pooling="average", **kwargs):
        super().__init__(config, add_pooling_layer=False)
        if not hasattr(config, "pooling"):
            self.config.pooling = pooling

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        last_hidden = model_output["last_hidden_state"]
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.0).clone()
        if self.config.pooling == "average":
            emb = last_hidden.sum(dim=1).clone() / attention_mask.sum(dim=1)[..., None].clone()
        elif self.config.pooling == "sqrt":
            emb = last_hidden.sum(dim=1) / torch.sqrt(attention_mask.sum(dim=1)[..., None].float())
        elif self.config.pooling == "cls":
            emb = last_hidden[:, 0]
        if normalize:
            emb = torch.nn.functional.normalize(emb, dim=-1).clone()
        return emb


class BaseRetriever(torch.nn.Module):
    """A retriever needs to be able to embed queries and passages, and have a forward function"""

    def __init__(self, *args, **kwargs):
        super(BaseRetriever, self).__init__()

    def embed_queries(self, *args, **kwargs):
        raise NotImplementedError()

    def embed_passages(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, is_passages=False, **kwargs):
        if is_passages:
            return self.embed_passages(*args, **kwargs)
        else:
            return self.embed_queries(*args, **kwargs)

    def gradient_checkpointing_enable(self):
        for m in self.children():
            m.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        for m in self.children():
            m.gradient_checkpointing_disable()


class DualEncoderRetriever(BaseRetriever):
    """Wrapper for standard contriever, or other dual encoders that parameter-share"""

    def __init__(self, opt, contriever):
        super(DualEncoderRetriever, self).__init__()
        self.opt = opt
        self.contriever = contriever

    def _embed(self, *args, **kwargs):
        return self.contriever(*args, **kwargs)

    def embed_queries(self, *args, **kwargs):
        return self._embed(*args, **kwargs)

    def embed_passages(self, *args, **kwargs):
        return self._embed(*args, **kwargs)




class UntiedDualEncoder(BaseRetriever):
    """Like DualEncoderRetriever, but dedicated encoders for passage and query embedding"""

    def __init__(self, opt, query_encoder, passage_encoder=None):
        """Create the module: if passage_encoder is none, one will be created as a deep copy of query_encoder"""
        super(UntiedDualEncoder, self).__init__()
        self.opt = opt
        self.query_contriever = query_encoder
        if passage_encoder is None:
            passage_encoder = copy.deepcopy(query_encoder) if hasattr(query_encoder, "module") else query_encoder
        self.passage_contriever = passage_encoder

    def embed_queries(self, *args, **kwargs):
        return self.query_contriever(*args, **kwargs)

    def embed_passages(self, *args, **kwargs):
        # if self.opt.query_side_retriever_training:
        #     is_train = self.passage_contriever.training
        #     self.passage_contriever.eval()
        #     with torch.no_grad():
        #         passage_emb = self.passage_contriever(*args, **kwargs)
        #     if is_train:
        #         self.passage_contriever.train()
        # else:
        passage_emb = self.passage_contriever(*args, **kwargs)

        return passage_emb
