# coding=utf-8
# Copyright 2022, The ATLAS Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""ATLAS model implementation."""

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union

import torch
from torch import nn
from functools import reduce


from ...configuration_utils import PretrainedConfig
from ...generation import BeamSearchScorer, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_atlas import AtlasConfig
from .tokenization_atlas import AtlasTokenizer

from .retriever import Contriever, UntiedDualEncoderRetriever, DualEncoderRetriever
from .fid import FiD


from ...utils import cached_file, is_datasets_available, is_faiss_available, logging, requires_backends

if is_datasets_available():
    from datasets import Dataset, load_dataset, load_from_disk

if is_faiss_available():
    import faiss

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "AtlasConfig"

class AtlasPreTrainedModel(PreTrainedModel):
    config_class = AtlasConfig
    base_model_prefix = "atlas"

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        # At the moment fast initialization is not supported
        # for composite models
        kwargs["_fast_init"] = False
        return super().from_pretrained(*args, **kwargs)
    
    @classmethod
    def from_pretrained_question_encoder_generator(
        cls,
        question_encoder_pretrained_model_name_or_path: str = None,
        generator_pretrained_model_name_or_path: str = None,
        index: Dataset = None,
        **kwargs
    ) -> PreTrainedModel:
        pass
        

class AtlasModel(AtlasPreTrainedModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        retriever: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        index: Optional[Dataset] = None,
        **kwargs,
    ):
        requires_backends(self, ["datasets", "faiss"])
        assert config is not None or (
            retriever is not None and generator is not None
        ), "Either a configuration or an retriever and a generator has to be provided."

        if config is None:
            config = AtlasConfig.from_pretrained_query_passage_generator(
                retriever.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        
        super().__init__(config)
        
        if retriever is None:
            from ..auto.modeling_auto import AutoModel
            contriever = Contriever(config.retriever)
            if config.query_side_retriever_training:
                retriever = UntiedDualEncoderRetriever(config, contriever)
            else:
                retriever = DualEncoderRetriever(config, contriever)

        if generator is None:
            from ..auto.modeling_auto import AutoModelForSeq2SeqLM
            generator = FiD(config.generator)

        self.index = index
        if self.index is not None:
            self.set_index(index)

        self.retriever = retriever
        self.generator = generator

    def set_index(self, dataset_with_index: Dataset):
        assert isinstance(
            dataset_with_index, Dataset
        ), f"`dataset_with_index` is of type {type(dataset_with_index)}, but should be of type `Dataset`"
        if len({"id", "text", "embeddings"} - set(dataset_with_index.column_names)) > 0:
            raise ValueError(
                "Dataset should be a dataset with the following columns: "
                "id (str), text (str) and embeddings (arrays of dimension vector_size), "
                f"but got columns {dataset_with_index.column_names}"
            )
        if "embeddings" not in dataset_with_index.list_indexes():
            raise ValueError(
                "Missing faiss index in the dataset. Make sure you called `dataset.add_faiss_index` to compute it "
                "or `dataset.load_faiss_index` to load one from the disk."
            )
        self.index = dataset_with_index
        self.index.set_format("numpy", columns=["embeddings"], output_all_columns=True, dtype="float32")
    
    def reindex(self, batch_size: int = 16):
        old_index = self.index.get_index("embeddings")
        device = old_index.device
        string_factory = old_index.string_factory
        metric_type = old_index.metric_type

        def reindex(examples):
            tokenized = self.tokenizer(examples['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)

            hidden_states = self.retriever.embed_passages(
                input_ids=tokenized["input_ids"].to(self.device),
                attention_mask=tokenized["attention_mask"].to(self.device)
            )
            examples['embeddings'] = hidden_states.cpu().detach().numpy()
            return examples

        self.index = self.index.index.map(reindex, batched=True, batch_size=batch_size)
        self.index.add_faiss_index("embeddings", device=device, string_factory=string_factory, metric_type=metric_type)

    # todo
    # - tokenize query with both tokenizers as input rather than in forward pass
    #    - this is especially ugly currently, as tokenizer might be on different device, shouldn't be us managing it
    # - make it possible to separate the retrieval and generation steps
    def forward(
        self,
        queries,
        target,
        target_tokens,
        topk,
    ):
        bsz = len(queries)

        queries_tokens = self.retriever_tokenizer(queries, return_tensors="pt", padding=True, truncation=True, max_length=512)

        query_hidden_states = self.retriever(input_ids=queries_tokens["input_ids"].to(self.device), attention_mask=queries_tokens["attention_mask"].to(self.device))

        query_hidden_states = query_hidden_states.cpu().detach().numpy()
        _, passage_ids = self.index.search_batch("embeddings", query_hidden_states, topk)
        docs = [self.index[[i for i in indices if i >= 0]] for indices in passage_ids]

        # todo: move this out to retriever - it's not the model's job to do this and it should be configurable
        passages = [[f'{queries[i]} context: {passage}' for passage in doc["text"]] for i, doc in enumerate(docs)]


        def encode_passages(batch, tokenizer, max_length):
            bsz = len(batch)
            n = max([len(example) for example in batch])
            batch = [example + [""] *  (n - len(example)) for example in batch]
            batch = reduce(lambda a, b: a + b, batch)
            tokens = tokenizer(
                batch,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
                truncation=True,
            )
            tokens = {k: v.view(bsz, n, -1) for k, v in tokens.items()}
            
            return tokens
        
        reader_tokens = encode_passages(passages, self.generator_tokenizer, 512)

        # <EXTRACT TO CALLER>
        labels = self.generator_tokenizer(target, return_tensors="pt", padding=True, truncation=True, max_length=512)['input_ids'].to(self.device)
        labels[labels == self.generator_tokenizer.pad_token_id] = -100
        # </EXTRACT TO CALLER>

        reader_ids = reader_tokens["input_ids"].to(self.device)  # FIXME (Not added by ae99, TODO figure out why)
        reader_mask = reader_tokens["attention_mask"].bool().to(self.device)


        # retriever_loss = None
        # if train_retriever:

        #     if self.opt.use_gradient_checkpoint_retriever:
        #         self.retriever.gradient_checkpointing_enable()

        #     query_emb = self.retriever(**query_enc, is_passages=False)

        #     retriever_tokens = {k: v.reshape(-1, v.size(-1)) for k, v in retriever_tokens.items()}

        #     passage_emb = self.retriever(**retriever_tokens, is_passages=True).to(query_emb)
        #     passage_emb = passage_emb.view(bsz, -1, passage_emb.size(-1))
        #     retriever_score = torch.einsum("id, ijd->ij", [query_emb, passage_emb])

        #     if self.opt.use_gradient_checkpoint_retriever:
        #         self.retriever.gradient_checkpointing_disable()

        #     gold_score = self.perplexity_score(reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz)

        #     if self.training:
        #         self.reader.train()


        n_context_training = min(topk, reader_ids.size(1))
        cfg = self.generator.encoder.config
        cfg.bsz = reader_ids.size(0)
        cfg.n_context = n_context_training

        reader_ids_training = reader_ids[:, :n_context_training].contiguous()
        reader_mask_training = reader_mask[:, :n_context_training].contiguous()

        reader_ids_training = reader_ids_training.view(reader_ids.size(0), -1)
        reader_mask_training = reader_mask_training.view(reader_mask.size(0), -1)

        generator_output = self.generator(
            input_ids=reader_ids_training,
            attention_mask=reader_mask_training,
            decoder_input_ids=None,
            labels=labels,
            use_cache=False,
        )

        reader_loss = generator_output[0]


        # if train_retriever:
        #     retriever_score = retriever_score / np.sqrt(query_emb.size(-1))

        #     if gold_score is not None:
        #         gold_score = gold_score.float()
        #         retriever_score = retriever_score.float()
    #             retriever_loss = self.kldivloss(retriever_score, gold_score)

        def kldivloss(self, score, gold_score):
            gold_score = torch.softmax(gold_score / self.opt.temperature_gold, dim=-1)
            score = torch.nn.functional.log_softmax(score / self.opt.temperature_score, dim=-1)
            return torch.nn.KLDivLoss()(score, gold_score)
    
    @torch.no_grad()
    def generate(self, tokens, query, choices=None):
        cfg = self.reader.encoder.config
        cfg.bsz = tokens["input_ids"].size(0)
        cfg.n_context = min(self.opt.n_context, tokens["input_ids"].size(1))

        tokens = {k: v.view(v.size(0), -1) for k, v in tokens.items()}

        bos_token_id = None

        prefix_allowed_tokens_fn = None
        if self.opt.decoder_prompt_format is not None:
            prefix_str = [self.opt.decoder_prompt_format.format_map({"query": q}) for q in query]
            prefix_allowed_tokens_fn = self.get_prefix_allowed_tokens_fn(prefix_str)

        outputs = self.generator.generate(
            input_ids=tokens["input_ids"].cuda(),
            attention_mask=tokens["attention_mask"].cuda(),
            num_return_sequences=1,
            max_length=self.opt.generation_max_length,
            min_length=self.opt.generation_min_length,
            num_beams=self.opt.generation_num_beams,
            length_penalty=self.opt.generation_length_penalty,
            forced_bos_token_id=bos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
        )

        return outputs

    def perplexity_score(self, reader_ids, reader_mask, decoder_input_ids, labels, cfg, bsz):
        with torch.no_grad():
            self.reader.eval()
            total_context = reader_ids.size(1)
            cfg.n_context = 1
            cfg.bsz = bsz * total_context
            reader_ids_score = reader_ids.view(bsz * total_context, -1)
            reader_mask_score = reader_mask.view(bsz * total_context, -1)
            repeated_decoder_input_ids = torch.repeat_interleave(decoder_input_ids, total_context, dim=0)
            repeated_labels = torch.repeat_interleave(labels, total_context, dim=0)
            reader_output = self.reader(
                input_ids=reader_ids_score.cuda(),
                attention_mask=reader_mask_score.cuda(),
                decoder_input_ids=repeated_decoder_input_ids,
                labels=repeated_labels,
                use_cache=False,
            )
            token_loss = nn.functional.cross_entropy(
                reader_output.logits.view(-1, reader_output.logits.size(-1)),
                repeated_labels.flatten(),
                reduction="none",
            )
            gold_score = token_loss.view(bsz, total_context, -1)
            z = (repeated_labels.view(bsz, total_context, -1) > -1).sum(dim=-1)
            gold_score = -gold_score.sum(dim=-1) / z

            return gold_score
