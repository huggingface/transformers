import os
from functools import partial
from glob import glob

from datasets import Features, Sequence, Value, concatenate_datasets, load_from_disk

import faiss
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast


def split_text(text, n=100, character=" "):
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    return [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]


def split_documents(documents):
    """Split documents into passages"""
    titles, texts = [], []
    for title, text in zip(documents["title"], documents["text"]):
        if text is not None:
            for passage in split_text(text):
                titles.append(title if title is not None else "")
                texts.append(passage)
    return {"title": titles, "text": texts}


def embed_update(ctx_encoder, context_tokenizer, device, process_num, data_shrad, shard_dir, data_cache_dir):

    arrow_folder = "data_" + str(process_num)
    arrow_file_name = "data_" + str(process_num) + ".arrow"

    passages_path = os.path.join(shard_dir, arrow_folder)  # shard_dir + arrow_folder+'/'
    cache_file_name = os.path.join(data_cache_dir, arrow_file_name)  # data_cache_dir + arrow_file_name

    ctx_encoder = ctx_encoder.to(device=device)
    # context_tokenizer=DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')

    def embed(
        documents: dict, ctx_encoder: DPRContextEncoder, ctx_tokenizer: DPRContextEncoderTokenizerFast, device
    ) -> dict:
        """Compute the DPR embeddings of document passages"""
        input_ids = ctx_tokenizer(
            documents["title"], documents["text"], truncation=True, padding="longest", return_tensors="pt"
        )["input_ids"]
        embeddings = ctx_encoder(input_ids.to(device=device), return_dict=True).pooler_output
        return {"embeddings": embeddings.detach().cpu().numpy()}

    new_features = Features(
        {"text": Value("string"), "title": Value("string"), "embeddings": Sequence(Value("float32"))}
    )  # optional, save as float32 instead of float64 to save space

    dataset = data_shrad.map(
        partial(embed, ctx_encoder=ctx_encoder, ctx_tokenizer=context_tokenizer, device=device),
        batched=True,
        batch_size=16,
        features=new_features,
        cache_file_name=cache_file_name,
        load_from_cache_file=False,
    )

    dataset.save_to_disk(passages_path)
    os.remove(dataset.cache_files[0]["filename"])


def add_index(shard_dir, index_path):
    data_shard_list = []

    for shard_address in glob(str(shard_dir) + "/*/"):
        data_shard_list.append(load_from_disk(shard_address))

    concat = concatenate_datasets(data_shard_list)
    faiss.omp_set_num_threads(96)

    index = faiss.IndexHNSWFlat(768, 128, faiss.METRIC_INNER_PRODUCT)
    concat.add_faiss_index("embeddings", custom_index=index)
    concat.get_index("embeddings").save(
        index_path
    )  # since we load the index in to memory,we can directly update the index
