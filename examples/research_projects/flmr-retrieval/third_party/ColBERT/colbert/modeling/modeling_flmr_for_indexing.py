"""
This file defines a simple wrapper around FLMRModelForRetrieval that can be directly used with the ColBERT engine for indexing.
Author: Weizhe Lin
Date: 01/02/2024
"""

from typing import Optional, Tuple, Union
from transformers import FLMRModelForRetrieval, FLMRConfig
from transformers import AutoImageProcessor
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm


def _sort_by_length(ids, mask, bsize, *args):
    if ids.size(0) <= bsize:
        return ids, mask, torch.arange(ids.size(0))

    indices = mask.sum(-1).sort().indices
    reverse_indices = indices.sort().indices
    
    return_array = [ids[indices], mask[indices]]
    for arg in args:
        if isinstance(arg, torch.Tensor):
            return_array.append(arg[indices])
        else:
            # arg is a list, and we want to sort the list according to indices
            return_array.append([arg[i] for i in indices])

    return *return_array, reverse_indices


def _split_into_batches(ids, mask, bsize, *args):
    batches = []
    for offset in range(0, ids.size(0), bsize):
        batch = [ids[offset:offset+bsize], mask[offset:offset+bsize]]
        for arg in args:
            batch.append(arg[offset:offset+bsize])
        batches.append(batch)
    return batches


def _stack_3D_tensors(groups):
    bsize = sum([x.size(0) for x in groups])
    maxlen = max([x.size(1) for x in groups])
    hdim = groups[0].size(2)

    output = torch.zeros(bsize, maxlen, hdim, device=groups[0].device, dtype=groups[0].dtype)

    offset = 0
    for x in groups:
        endpos = offset + x.size(0)
        output[offset:endpos, :x.size(1)] = x
        offset = endpos

    return output


class FLMRModelForIndexing(FLMRModelForRetrieval):
    
    def __init__(
            self, config: FLMRConfig, 
            **kwargs,
        ):
        super().__init__(config, **kwargs)
        self.image_processor = AutoImageProcessor.from_pretrained(self.config.vision_model_version)
    
    def query(self, *args, to_cpu=False, **kw_args):
        Q = super().query(*args, **kw_args)
        return Q.cpu() if to_cpu else Q

    def doc(self, *args, to_cpu=False, **kw_args):
        D = super().doc(*args, **kw_args)

        if to_cpu:
            return (D[0].cpu(), *D[1:]) if isinstance(D, tuple) else D.cpu()

        return D

    def queryFromText(self, queries, bsize=None, to_cpu=False, context=None):
        if bsize:
            batches = self.query_tokenizer(queries, context=context, bsize=bsize)
            batches = [self.query(input_ids, attention_mask, to_cpu=to_cpu) for input_ids, attention_mask in batches]
            batches = [b.late_interaction_output for b in batches]
            return torch.cat(batches)

        input_ids, attention_mask = self.query_tokenizer(queries, context=context)
        return self.query(input_ids, attention_mask)

    def docFromText(self, docs, bsize=None, keep_dims=True, to_cpu=False, showprogress=False, return_tokens=False):
        assert keep_dims in [True, False, 'flatten']

        # docs can be
        # (1) list of text
        # (2) list of tuples (text, image_features, None)
        # (3) list of tuples (text, None, image_paths)

        if isinstance(docs[0], tuple):
            texts = []
            image_features = []
            image_paths = []
            for doc in docs:
                text, image_feature, image_path = doc
                texts.append(text)
                image_features.append(image_feature)
                image_paths.append(image_path)
            
            docs = texts
            if image_features[0] is not None:
                image_features = torch.FloatTensor(np.stack(image_features))
                is_input_image_features = True
            else:
                is_input_image_features = False

            multimodal_docs = True
        else:
            image_features = None
            image_paths = None
            multimodal_docs = False

        if bsize:
            # we change this part to enable dynamically loading image features to avoid memory overflow
            # This bsize function is used in the original ColBERT codebase to split inputs into multiple batches
            context_encoding = self.context_tokenizer(docs)
            ids, mask = context_encoding['input_ids'], context_encoding['attention_mask']
            
            if multimodal_docs:
                # print(ids[0], mask[0], image_features[0], image_paths[0])
                # print(image_features.shape)
                ids, mask, image_features, image_paths, reverse_indices = _sort_by_length(ids, mask, bsize, image_features, image_paths)
                # print(image_features.shape)
                # print(len(ids), len(mask), len(image_features), len(image_paths))
                # print(ids[0], mask[0], image_features[0], image_paths[0])
                batches = _split_into_batches(ids, mask, bsize, image_features, image_paths)
            else:
                ids, mask, reverse_indices = _sort_by_length(ids, mask, bsize)
                batches = _split_into_batches(ids, mask, bsize)

            # text_batches, reverse_indices = self.context_tokenizer(docs, bsize=bsize)
            
            returned_text = []
            if return_tokens:
                text_batches = [(input_ids, attention_mask) for input_ids, attention_mask, _, _ in batches]
                returned_text = [text for batch in text_batches for text in batch[0]]
                returned_text = [returned_text[idx] for idx in reverse_indices.tolist()]
                returned_text = [returned_text]
            
            
            keep_dims_ = True if keep_dims == 'flatten' else keep_dims
            return_mask = True if keep_dims == 'flatten' else False

            encoded_batches = []

            for batch in tqdm(batches, disable=not showprogress):
                if multimodal_docs:
                    input_ids, attention_mask, image_features, image_paths = batch
                    if is_input_image_features:
                        context_output = self.doc(
                            input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            image_features=image_features,
                            keep_dims=keep_dims_, 
                            return_mask=return_mask, 
                            to_cpu=to_cpu,
                        )
                    else:
                        # Open the images in image_paths and convert to pixel_values by using ImageProcessor
                        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
                        pixel_values = self.image_processor(images, return_tensors="pt").pixel_values
                        print(pixel_values.shape)
                        context_output = self.doc(
                            input_ids, 
                            attention_mask, 
                            pixel_values=pixel_values, 
                            keep_dims=keep_dims_, 
                            return_mask=return_mask, 
                            to_cpu=to_cpu,
                        )
                else:
                    input_ids, attention_mask = batch
                    context_output = self.doc(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        keep_dims=keep_dims_, 
                        return_mask=return_mask, 
                        to_cpu=to_cpu,
                    )
                encoded_batches.append(context_output)
                
            
            if keep_dims is True:
                D = _stack_3D_tensors(batches)
                return (D[reverse_indices], *returned_text)

            elif keep_dims == 'flatten':
                D, mask = [], []

                for batch in encoded_batches:
                    D_, mask_ = batch.late_interaction_output, batch.context_mask
                    D.append(D_)
                    mask.append(mask_)

                D, mask = torch.cat(D)[reverse_indices], torch.cat(mask)[reverse_indices]

                doclens = mask.squeeze(-1).sum(-1).tolist()

                D = D.view(-1, self.config.dim)
                D = D[mask.bool().flatten()].cpu()

                return (D, doclens, *returned_text)

            assert keep_dims is False

            D = [d for batch in batches for d in batch]
            return ([D[idx] for idx in reverse_indices.tolist()], *returned_text)

        input_ids, attention_mask = self.context_tokenizer(docs)
        return self.doc(input_ids, attention_mask, keep_dims=keep_dims, to_cpu=to_cpu)
