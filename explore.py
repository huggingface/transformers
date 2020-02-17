__author__ = 'qianchu_liu'

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
from sklearn.metrics.pairwise import cosine_similarity
import argparse

import logging
logging.basicConfig(level=logging.INFO)


def preprocess_text(text_f,tokenizer):
    indexed_tokens_lst=[]
    segments_ids_lst=[]
    orig_to_token_map_lst=[]
    for text in open(text_f):
        text=text.strip().split()
        tokenized_text,orig_to_token_map = tokenize_map(text,tokenizer)
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [0] * len(indexed_tokens)
        indexed_tokens_lst.append(indexed_tokens)
        segments_ids_lst.append(segments_ids)
        orig_to_token_map_lst.append(orig_to_token_map)
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])
    tokens_tensor = torch.tensor(indexed_tokens_lst)
    segments_tensors = torch.tensor(segments_ids_lst)
    return tokens_tensor,segments_tensors,orig_to_token_map_lst




def feature_extract_batch(text_f, tokenizer,model):

    ###tokenize text
    # text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokens_tensor, segments_tensors,orig_to_token_map_batch=preprocess_text(text_f,tokenizer)

    # segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12
    average_layer_batch=sum(encoded_layers[-4:])/4
    # if orig_to_token_map_batch!=None:
    average_layer_batch=feature_orig_to_tok_map(average_layer_batch,orig_to_token_map_batch)
    return average_layer_batch



def tokenize_map(orig_tokens,tokenizer):
    ### Input
    labels = ["NNP", "NNP", "POS", "NN"]

    ### Output
    bert_tokens = []

    # Token map will be an int -> int mapping between the `orig_tokens` index and
    # the `bert_tokens` index.
    orig_to_tok_map = []

    bert_tokens.append("[CLS]")
    for orig_token in orig_tokens:
        orig_to_tok_map.append(len(bert_tokens))
        bert_tokens.extend(tokenizer.tokenize(orig_token))
    bert_tokens.append("[SEP]")
    return bert_tokens,orig_to_tok_map


def feature_orig_to_tok_map(average_layer_batch, orig_to_token_map_batch):
    average_layer_batch_out=[]
    for sent_embed in average_layer_batch:
        sent_embed_out=[]
        for i in range(len(orig_to_token_map_batch)):
            start = orig_to_token_map_batch[i]
            end = orig_to_token_map_batch[i + 1]
            if i==(len(orig_to_token_map_batch)-1):
                sent_embed_out.append(sum(sent_embed[start:len(sent_embed)-1]) / (len(sent_embed)-1 - start))
            sent_embed_out.append(sum(sent_embed[start:end])/(end-start))
        average_layer_batch_out.append(sent_embed_out)
    return average_layer_batch_out


def feature_extract(text, tokenizer,model,word_indexes=None):

    ###tokenize text
    # text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = tokenizer.tokenize(text)
    print (tokenized_text)

    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_ids=[0]*len(indexed_tokens)
    segments_tensors = torch.tensor([segments_ids])

    # segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]


    model.eval()

    # Predict hidden states features for each layer
    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)
    # We have a hidden states for each of the 12 layers in model bert-base-uncased
    assert len(encoded_layers) == 12

    average_layer=sum(encoded_layers[-4:])/4
    average_layer=average_layer[0]
    if word_indexes!=None:
        average_layer=sum(average_layer[word_indexes[0]:word_indexes[1]])/(word_indexes[1]-word_indexes[0])

    return average_layer

def vocab_embed(probe_vocab,tokenizer,model):
    vocab_embeds=[]
    for vocab in probe_vocab:
        vocab='[CLS] '+vocab
        vocab_embed=sum(feature_extract(vocab,tokenizer,model)[1:]).cpu().detach().numpy()
        # vocab_embed=feature_extract(vocab,tokenizer,model)[0].cpu().detach().numpy()

        print (vocab_embed.shape)
        vocab_embeds.append(vocab_embed)
    print(cosine_similarity(vocab_embeds,vocab_embeds))
    return vocab_embeds

# def preprocess_text(text):
#
#     for line in open(text):

if __name__=='__main__':
    args = argparse.ArgumentParser('extract bert embed')
    args.add_argument('--pretrained',  type=str, help='pretrained model')
    args.add_argument('--text', type=str, help='text file: one sentence per line')
    args=args.parse_args()

    text="[CLS] 我 喜歡 吃 [MASK] 。 [SEP]"
    probe_vocab=['難過','快樂','開心','開門','香蕉','西紅柿']
    tokenizer = BertTokenizer.from_pretrained(args.pretrained)
    model = BertModel.from_pretrained(args.pretrained)

    in_context=feature_extract(text,tokenizer,model,(4,5)).cpu().detach().numpy()

    vocab_embeds=vocab_embed(probe_vocab,tokenizer,model)
    print(cosine_similarity([in_context],vocab_embeds))
    # features=feature_extract_batch(args.text,tokenizer,model)