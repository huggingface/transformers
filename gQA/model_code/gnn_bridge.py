import string
import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


from gQA.model_code.gnn import *
from gQA.model_code.sentence_selector import Sentence_Selector
from gQA.model_code.data_opinosis import *
from gQA.model_code.state import State

def isnan(x):
    return x != x
#Acts as a bridge between the encoder and decoder
class GNN_Bridge(nn.Module):
    def __init__(self, hidden_size, max_sentence_len, config, sample_size = 16):
        super(GNN_Bridge, self).__init__()
        self.sample_size = sample_size
        self.hidden_size = hidden_size
        self.d_graph = [self.hidden_size]
        self.gnn_layer = nn.ModuleList()

        #Selects which sentences to attend over
        self.sentence_selector = Sentence_Selector(hidden_size, config)

        d_in = self.hidden_size
        i = 1
        for d_out in self.d_graph:
            self.gnn_layer.append(GNN_Layer(d_in, d_out, globalnode=False))
            d_in = d_out
            i += 1
            
        self.MAX_SENTENCE_LEN = max_sentence_len

    def forward(self, context, question=None):

        #Unpack input
        adj, dev, dimensions = context.other
        batch_size, docu_len, sent_len = dimensions
        h_sent = context.outputs

        #GNN
        #Nothing special going on here. Read graphIE sentence-level's model to understand this approach
        #The code for GNNs was directly ripped from graphIE, with some small changes.
        #(e.g. masked mean was removed, it seems as if the original implementation was erroneous)
        h_sent = h_sent.view(batch_size, docu_len, -1)

        h_gcn = None
        h_gcn = h_sent
        gcn_layers = []
        #Append the actual sentence embeddings
        gcn_layers.append(h_gcn)
        '''
        #Append the document representation, but only if we are using GCNs
        if self.n_mult >= 1:
            gcn_layers.append(self.doc_representation(h_gcn, all_to_all))
        '''

        #Append the propogated sentence embeddings
        for i in range(len(self.gnn_layer)):
            h_gcn = self.gnn_layer[i](h_gcn, adj)
            gcn_layers.append(h_gcn)

        gcn_layers_stacked = torch.stack(gcn_layers)
        # gcn_layers: num layeres, batch, max article len, sent dim
        h_gcn = h_gcn.to(dev)
        
        return self.sentence_selector(
            context.mask,
            gcn_layers=gcn_layers_stacked,
            gcn_raw=gcn_layers,
            dimensions=[batch_size, docu_len, sent_len]), h_gcn