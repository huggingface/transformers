import logging
import random

import ray
from transformers import RagConfig, RagRetriever, RagTokenizer
from transformers.file_utils import requires_datasets, requires_faiss
from transformers.models.rag.retrieval_rag import CustomHFIndex


from transformers import (
    DPRContextEncoder,
    DPRConfig,
    DPRContextEncoderTokenizerFast,
    HfArgumentParser
)

config_dpr=DPRConfig.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')


import torch

import faiss
import os

from datasets import Features, Sequence, Value, load_dataset,load_from_disk
from functools import partial
import glob

logger = logging.getLogger(__name__)


class RayEncode:
    def __init__(self):
        
        self.initialized=False
        print("This actor is allowed to use GPUs {}.".format(ray.get_gpu_ids()))
        self.device = "cuda:2" #I wanted to make sure the this saving Ray worker only use the remaining GPU. 


    def load_utils(self, config, context_encoder_tokenizer,dataset):
        
        self.context_encoder_tokenizer=context_encoder_tokenizer
        #each worker can load already assiged dataset
        self.dataset = dataset
        self.config=config       

    #@ray.remote(num_gpus=1, max_calls=1)
    #@amogkam I tried to use max_calls, but then the programme stucked.
    def save_indexed_embeddings(self,ctx_encoder):
        
        #the saving code was taken from the use_own_knowledge_dataset.py
        new_features = Features(
            {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
        )  # optional, save as float32 instead of float64 to save space

        self.dataset = self.dataset.map(
            partial(self.embed, ctx_encoder=ctx_encoder.to(device=self.device), ctx_tokenizer=self.context_encoder_tokenizer),
            batched=True,
            batch_size=8,
            features=new_features,
        )

        passages_path =self.config.passages_path
        files = glob.glob(os.path.join(passages_path+'/*'))

        for file in files:
            os.remove(file)        

    
        self.dataset.save_to_disk(passages_path)

        # Let's use the Faiss implementation of HNSW for fast approximate nearest neighbor search
        index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
        self.dataset.add_faiss_index("embeddings", custom_index=index)

        # And save the index
        index_path=self.config.index_path
        self.dataset.get_index("embeddings").save(index_path)
       


    def embed(self,documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast):
        """Compute the DPR embeddings of document passages"""
        input_ids = ctx_tokenizer(
            documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
        )["input_ids"]

        embeddings = ctx_encoder(input_ids.to(device=self.device), return_dict=True).pooler_output

        return {"embeddings": embeddings.detach().cpu().numpy()}


class RayCreateEmbedding:

    def __init__(self, encode_worker,context_encoder_tokenizer,config):


        self.encode_workers =encode_worker
        self.context_encoder_tokenizer=context_encoder_tokenizer

        #here I used already split dataset which was saved as a huggiface dataset object.
        self.dataset=load_from_disk(config.csv_path)

        #later can load the self.dataset directly
        if len( self.encode_workers) > 0: #initialize rag_retriver in each of the class
            ray.get(
                [
                    worker.load_utils.remote(config,self.context_encoder_tokenizer,self.dataset)
                    for worker in self.encode_workers
                ]
            )

    # @amogkam this function get the updated ctx_encoder weights and get the new passages with index.
    def save_indexed_embeddings(self,ctx_encoder):  

        logger.info("encoding and saving the embeddings with updated ctx_encoder")

        model_copy =type(ctx_encoder)(config_dpr) # get a new instance
        model_copy.load_state_dict(ctx_encoder.state_dict()) # copy weights and stuff
        new_ctx=model_copy

    
        if len(self.encode_workers) > 0:
            ray.get([worker.save_indexed_embeddings.remote(new_ctx) for worker in self.encode_workers])

