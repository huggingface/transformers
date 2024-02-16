"""
This script is an example of how to use the pretrained FLMR model for retrieval.
Author: Weizhe Lin
Date: 31/01/2024
For more information, please refer to the official repository of FLMR:
https://github.com/LinWeizheDragon/Retrieval-Augmented-Visual-Question-Answering
"""

import os
from collections import defaultdict

import numpy as np
import torch
from colbert import Indexer, Searcher
from colbert.data import Queries
from colbert.infra import ColBERTConfig, Run, RunConfig
from easydict import EasyDict
from PIL import Image

from transformers import (
    AutoImageProcessor,
    FLMRContextEncoderTokenizer,
    FLMRModelForRetrieval,
    FLMRQueryEncoderTokenizer,
)


def index_corpus(args, custom_collection):
    # Launch indexer
    with Run().context(RunConfig(nranks=1, root=args.index_root_path, experiment=args.experiment_name)):
        config = ColBERTConfig(
            nbits=args.nbits,
            doc_maxlen=512,
            total_visible_gpus=1 if args.use_gpu else 0,
        )
        print("indexing with", args.nbits, "bits")

        indexer = Indexer(checkpoint=args.checkpoint_path, config=config)
        indexer.index(
            name=f"{args.index_name}.nbits={args.nbits}",
            collection=custom_collection,
            batch_size=args.indexing_batch_size,
            overwrite=True,
        )
        index_path = indexer.get_index()

        return index_path


def query_index(args, ds, passage_contents, flmr_model: FLMRModelForRetrieval):
    # Search documents
    with Run().context(RunConfig(nranks=1, rank=1, root=args.index_root_path, experiment=args.experiment_name)):
        if args.use_gpu:
            total_visible_gpus = torch.cuda.device_count()
        else:
            total_visible_gpus = 0

        config = ColBERTConfig(
            total_visible_gpus=total_visible_gpus,
        )
        searcher = Searcher(
            index=f"{args.index_name}.nbits={args.nbits}", checkpoint=args.checkpoint_path, config=config
        )

        def encode_and_search_batch(batch, Ks):
            # encode queries
            input_ids = torch.LongTensor(batch["input_ids"]).to("cuda")
            # print(query_tokenizer.batch_decode(input_ids, skip_special_tokens=False))
            attention_mask = torch.LongTensor(batch["attention_mask"]).to("cuda")
            pixel_values = torch.FloatTensor(batch["pixel_values"]).to("cuda")
            query_input = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
            query_embeddings = flmr_model.query(**query_input).late_interaction_output
            query_embeddings = query_embeddings.detach().cpu()

            # search
            custom_quries = {
                question_id: question for question_id, question in zip(batch["question_id"], batch["question"])
            }
            # print(custom_quries)
            queries = Queries(data=custom_quries)
            ranking = searcher._search_all_Q(
                queries,
                query_embeddings,
                progress=False,
                batch_size=args.centroid_search_batch_size,
                k=max(Ks),
                remove_zero_tensors=True,  # For PreFLMR, this is needed
            )

            ranking_dict = ranking.todict()

            # Process ranking data and obtain recall scores
            recall_dict = defaultdict(list)
            for question_id, answers in zip(batch["question_id"], batch["answers"]):
                retrieved_docs = ranking_dict[question_id]
                retrieved_docs = [doc[0] for doc in retrieved_docs]
                retrieved_doc_texts = [passage_contents[doc_idx] for doc_idx in retrieved_docs]
                hit_list = []
                for retrieved_doc_text in retrieved_doc_texts:
                    found = False
                    for answer in answers:
                        if answer.strip().lower() in retrieved_doc_text.lower():
                            found = True
                    if found:
                        hit_list.append(1)
                    else:
                        hit_list.append(0)

                # print(hit_list)
                # input()
                for K in Ks:
                    recall = float(np.max(np.array(hit_list[:K])))
                    recall_dict[f"Recall@{K}"].append(recall)

            batch.update(recall_dict)
            return batch

        flmr_model = flmr_model.to("cuda")
        print("Starting encoding...")
        Ks = args.Ks
        # ds = ds.select(range(2000, 2100))
        ds = ds.map(
            encode_and_search_batch,
            fn_kwargs={"Ks": Ks},
            batched=True,
            batch_size=args.query_batch_size,
            load_from_cache_file=False,
            new_fingerprint="avoid_cache",
        )

        return ds


def main(args):
    from datasets import load_dataset

    ds = load_dataset(args.dataset_path)
    passage_ds = load_dataset(args.passage_dataset_path)

    print("========= Loading dataset =========")
    print(ds)
    print(passage_ds)

    def add_path_prefix_in_img_path(example, prefix):
        example["img_path"] = os.path.join(prefix, example["img_path"])
        return example

    ds = ds.map(add_path_prefix_in_img_path, fn_kwargs={"prefix": args.image_root_dir})

    use_split = args.use_split

    ds = ds[use_split]
    passage_ds = passage_ds[f"{use_split}_passages"]
    print("========= Data Summary =========")
    print("Number of examples:", len(ds))
    print("Number of passages:", len(passage_ds))

    print("========= Indexing =========")
    # Run indexing on passages
    passage_contents = passage_ds["passage_content"]
    # passage_contents =['<BOK> ' + passage + ' <EOK>' for passage in passage_contents]

    if args.run_indexing:
        ## Call ColBERT indexing to index passages
        index_corpus(args, passage_contents)
    else:
        print("args.run_indexing is False, skipping indexing...")

    print("========= Loading pretrained model =========")
    query_tokenizer = FLMRQueryEncoderTokenizer.from_pretrained(args.checkpoint_path, subfolder="query_tokenizer")
    context_tokenizer = FLMRContextEncoderTokenizer.from_pretrained(
        args.checkpoint_path, subfolder="context_tokenizer"
    )

    flmr_model = FLMRModelForRetrieval.from_pretrained(
        args.checkpoint_path,
        query_tokenizer=query_tokenizer,
        context_tokenizer=context_tokenizer,
    )
    image_processor = AutoImageProcessor.from_pretrained(args.image_processor_name)

    print("========= Preparing query input =========")

    instructions = [
        "Using the provided image, obtain documents that address the subsequent question: ",
        "Retrieve documents that provide an answer to the question alongside the image: ",
        "Extract documents linked to the question provided in conjunction with the image: ",
        "Utilizing the given image, obtain documents that respond to the following question: ",
        "Using the given image, access documents that provide insights into the following question: ",
        "Obtain documents that correspond to the inquiry alongside the provided image: ",
        "With the provided image, gather documents that offer a solution to the question: ",
        "Utilizing the given image, obtain documents that respond to the following question: ",
    ]
    import random

    def prepare_inputs(sample):
        sample = EasyDict(sample)

        module = EasyDict(
            {"type": "QuestionInput", "option": "default", "separation_tokens": {"start": "", "end": ""}}
        )

        random_instruction = random.choice(instructions)
        text_sequence = " ".join(
            [random_instruction]
            + [module.separation_tokens.start]
            + [sample.question]
            + [module.separation_tokens.end]
        )

        sample["text_sequence"] = text_sequence

        return sample

    # Prepare inputs using the same configuration as in the original FLMR paper
    ds = ds.map(prepare_inputs)

    def tokenize_inputs(examples, query_tokenizer, image_processor):
        encoding = query_tokenizer(examples["text_sequence"])
        examples["input_ids"] = encoding["input_ids"]
        examples["attention_mask"] = encoding["attention_mask"]

        pixel_values = []
        for img_path in examples["img_path"]:
            image = Image.open(img_path).convert("RGB")
            encoded = image_processor(image, return_tensors="pt")
            pixel_values.append(encoded.pixel_values)

        pixel_values = torch.stack(pixel_values, dim=0)
        examples["pixel_values"] = pixel_values
        return examples

    # Tokenize and prepare image pixels for input
    ds = ds.map(
        tokenize_inputs,
        fn_kwargs={"query_tokenizer": query_tokenizer, "image_processor": image_processor},
        batched=True,
        batch_size=8,
        num_proc=16,
    )

    print("========= Querying =========")
    ds = query_index(args, ds, passage_contents, flmr_model)
    # Compute final recall
    print("=============================")
    print("Inference summary:")
    print("=============================")
    print(f"Total number of questions: {len(ds)}")

    for K in args.Ks:
        recall = np.mean(np.array(ds[f"Recall@{K}"]))
        print(f"Recall@{K}:\t", recall)

    print("=============================")
    print("Done! Program exiting...")


if __name__ == "__main__":
    # Initialize arg parser
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--use_gpu", action="store_true")
    # all hardcode parameters should be here
    parser.add_argument("--query_batch_size", type=int, default=8)
    parser.add_argument("--num_ROIs", type=int, default=9)
    parser.add_argument("--dataset_path", type=str, default="OKVQA_FLMR_prepared_data.hf")
    parser.add_argument(
        "--passage_dataset_path", type=str, default="OKVQA_FLMR_prepared_passages_with_GoogleSearch_corpus.hf"
    )
    parser.add_argument("--image_root_dir", type=str, default="./ok-vqa/")
    parser.add_argument("--use_split", type=str, default="test")
    parser.add_argument("--index_root_path", type=str, default=".")
    parser.add_argument("--index_name", type=str, default="OKVQA_GS")
    parser.add_argument("--experiment_name", type=str, default="OKVQA_GS")
    parser.add_argument("--indexing_batch_size", type=int, default=64)
    parser.add_argument("--image_processor_name", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--Ks", type=int, nargs="+", default=[5, 10, 20, 50, 100])
    parser.add_argument("--checkpoint_path", type=str, default="./converted_flmr")
    parser.add_argument("--run_indexing", action="store_true")
    parser.add_argument("--centroid_search_batch_size", type=int, default=None)

    args = parser.parse_args()
    """
    Example usage:
    python example_use_preflmr.py \
            --use_gpu --run_indexing \
            --index_root_path "." \
            --index_name EVQA_PreFLMR_ViT-L \
            --experiment_name EVQA \
            --indexing_batch_size 64 \
            --image_root_dir /path/to/EVQA/eval_image/ \
            --dataset_path BByrneLab/EVQA_PreFLMR_preprocessed_data \
            --passage_dataset_path BByrneLab/EVQA_PreFLMR_preprocessed_passages \
            --use_split test \
            --nbits 8 \
            --Ks 1 5 10 20 50 100 \
            --checkpoint_path LinWeizheDragon/PreFLMR_ViT-L \
            --image_processor_name openai/clip-vit-large-patch14 \
            --query_batch_size 8 \
    """
    main(args)
