import string
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from gQA.model_code.gnn import *
from gQA.model_code.data_opinosis import *
from transformers.modeling_bert import BertAttention
#Determines which sentences we want to sample for the decoder. Essentially performing an
#extractive summary before performing an abstractive one
class Sentence_Selector(nn.Module):

    def __init__(self, hidden_size, config):
        super(Sentence_Selector, self).__init__()
        self.hidden_size = hidden_size
        hs = hidden_size
        d_hs = hidden_size + 0

        self.attn_linear_base = nn.Linear(hs, d_hs)
        self.attn_linear_seq = nn.Linear(hs, d_hs)
        self.attn_linear_query = nn.Linear(hs, d_hs)
        self.sentence_attn = BertAttention(config)
        
    def compute_attn_inp(self, gcn_raw, batch_size, docu_len):
        #Combine GCN layers
        #base = self.attn_linear_base(gcn_raw[0].view(-1, self.hidden_size))
        neighbours = gcn_raw[1].view(-1, self.hidden_size)
        '''
        salience = self.attn_linear_all_to_all(gcn_raw[2].view(-1, self.hidden_size))
        '''
        query = gcn_raw[-1].view(-1, self.hidden_size)

        output = neighbours + query
        return output.view(batch_size, docu_len, -1)
    
    def forward(self, mask, gcn_layers=None, gcn_raw=None, dimensions=[0, 0, 0]):
        
        #Unpack dimensions
        batch_size, docu_len, _ = dimensions

        #Convert to a format that is best suited for BERT
        mask = (1 - mask.squeeze()) * -10000
        mask = mask.view(batch_size, 1, 1, -1)

        #Apply attention to the concat of the sentence and enrichments
        gcn_layers = gcn_layers.view(batch_size, docu_len, -1)

        #Compute attn vec
        attn_inp = self.compute_attn_inp(gcn_raw, batch_size, docu_len)
        decoder_hidden = self.sentence_attn(attn_inp, attention_mask=mask, query=None)[0]

        #Apply attention elsewhere
        #decoder_hidden = torch.bmm(torch.transpose(gcn_raw[1], 1, 2), sent_attn.unsqueeze(2)).squeeze(2)

        return decoder_hidden