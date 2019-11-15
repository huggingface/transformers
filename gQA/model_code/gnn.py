import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import string
import os
import json
import time
import sys
from gQA.model_code.ModelUtils import RNN, CRF
from gQA.model_code.attention import MultiHeadAttention, PositionAwareAttention
#Required by BERT
MAX_SENTENCE_LEN = 20

#Pulled directly from graphIE
class Character_CNN(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes):
        super(Character_CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=k, padding=k-1) 
            for k in filter_sizes])

    def _sort_tensor(self, input, mask):
        ''' 
        Returns the sorted tensor, the sorted seq length, and the indices for inverting the order.
        Input:
                input: batch_size, *
                lengths: batch_size
        Output:
                sorted_tensor: batch_size-num_zero, *
                sorted_len:    batch_size-num_zero
                sorted_order:  batch_size
                num_zero
        '''
        sorted_mask, sorted_order = mask.sort(0, descending=True)
        sorted_input = input[sorted_order]
        _, invert_order  = sorted_order.sort(0, descending=False)

        # Calculate the num. of sequences that have len 0
        nonzero_idx = sorted_mask.nonzero()
        num_nonzero = nonzero_idx.size()[0]
        num_zero = sorted_mask.size()[0] - num_nonzero

        # temporarily remove seq with len zero
        sorted_input = sorted_input[:num_nonzero]
        sorted_mask = sorted_mask[:num_nonzero]

        return sorted_input, sorted_mask, invert_order, num_zero

    def _unsort_tensor(self, input, invert_order, num_zero):
        ''' 
        Recover the origin order
        Input:
                input:        batch_size-num_zero, *
                invert_order: batch_size
                num_zero  
        Output:
                out:   batch_size, *
        '''
        if num_zero == 0:
            input = input.index_select(0, invert_order)

        else:
            zero = torch.zeros(num_zero, *(input.size()[1:]))
            zero = Variable(zero)
            zero = zero.cuda()

            input = torch.cat((input, zero), dim=0)
            input = input.index_select(0, invert_order)

        return input

    def forward(self, x, mask=None):
        if mask is not None:
            x, sorted_mask, invert_order, num_zero = self._sort_tensor(x, mask)
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        x = [torch.tanh(conv(x)) for conv in self.convs]  # [**, num_filters, **]
        x = [F.max_pool1d(sub_x, sub_x.size(2)).squeeze(2) for sub_x in x]  # [**, num_filters]
        x = torch.cat(x, 1)
        if mask is not None:
            x = self._unsort_tensor(x, invert_order, num_zero)
        return x


class Word_RNN(nn.Module):

    def __init__(self, d_input, d_output, cell_type='LSTM', bi=True):
        super(Word_RNN, self).__init__()
        self.cell_type = cell_type
        self.lstm = RNN(d_input=d_input, d_hidden=d_output, 
        n_layers=1, cell_type=cell_type, pooling='mean', bidirectional=bi)

        # self.lstm = nn.LSTM(input_dim, output_dim, batch_first=True, bidirectional=False)

    def forward(self, text, text_len, text_mask, initial=None):
        if initial is not None:
            batch, d_hidden = initial.size()
            initial = initial.view(batch, 2, d_hidden//2)
        output, hidden = self.lstm(text, text_len, text_mask, initial)
        return output, hidden
    def clear_time(self):
        self.lstm.clear_time()

class AttnDecoder_RNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, 
    max_length=MAX_SENTENCE_LEN, sample_size = 3, dmult=2, gdim = 1):
        super(AttnDecoder_RNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length
        self.sample_size = sample_size
        self.gdim = gdim

        #Used for coverage. Both of these are unused currently.
        self.context_combine = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        #Attn mechanism
        self.encoder_reduce = nn.Linear(self.hidden_size, self.hidden_size * dmult)
        self.embed_reduce = nn.Linear(self.hidden_size, self.hidden_size * dmult)
        self.hidden_reduce = nn.Linear(self.hidden_size * dmult, self.hidden_size * dmult)

        self.attn = nn.Linear(self.hidden_size * dmult, 1)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        #Standard decoder stuff
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out1 = nn.Linear(self.hidden_size + self.hidden_size, self.hidden_size)
        self.out2 = nn.Linear(self.hidden_size, output_size)
        self.time = int(round(time.time() * 1000))

        #Pointer generation stuff
        self.p_gen_linear = nn.Linear(hidden_size * 2 + hidden_size * dmult, 1)
        self.d_mult = dmult

    def forward(self, embedded, s_t, encoder_outputs,
    coverage, mask=None, pointer_generation=True,
    extra_zeros=None, extended=None, is_coverage=False, sample_attn=None):
        embedded = self.dropout(embedded)
        batch_size, t_k, n = encoder_outputs.size()

        h_decoder, c_decoder = s_t
        s_t_hat = torch.cat((h_decoder.view(-1, n), 
        c_decoder.view(-1, n)), 1)  # B x 2*hidden_dim
        embedded = embedded.expand(self.gdim, batch_size, n//self.gdim).contiguous().view(1, batch_size, n)
        #Perform most of our resizes
        encoder_outputs = encoder_outputs.view(-1, n) #B * t_k x n

        attn_embedded = embedded[0].unsqueeze(1).expand(batch_size, t_k, n).contiguous().view(-1, n)
        attn_hidden = s_t_hat.unsqueeze(1).expand(batch_size, t_k, n * 2)
        attn_hidden = attn_hidden.contiguous().view(-1, n * 2)
        
        #Perform our attention computations
        attn_inp = self.encoder_reduce(encoder_outputs) \
        + self.embed_reduce(attn_embedded) \
        + self.hidden_reduce(attn_hidden) \

        attn_inp = torch.tanh(attn_inp)
        attn_weights = self.attn(attn_inp)
        attn_weights = attn_weights.view(batch_size, -1)
        
        encoder_outputs = encoder_outputs.view(batch_size, -1, n) #B x t_k x n

        #Mask over our attention
        if mask is not None:
            assert mask.size() == attn_weights.size()
            mask = mask.eq(0).data

            attn_weights.data.masked_fill_(mask, -float('inf'))
            attn_weights = self.dropout(self.softmax(attn_weights))

            attn_weights = attn_weights #* sample_attn
            
            
        #Apply attention to the encoder outputs
        attn_applied = torch.bmm(torch.transpose(encoder_outputs, 1, 2), attn_weights.unsqueeze(2))
        attn_applied = torch.transpose(attn_applied, 0, 2)
        attn_applied = torch.transpose(attn_applied, 1, 2)

        #Compute context
        context = attn_applied.view(batch_size, -1)

        #Compute p_gen probability using the prior context, embedding, and hidden
        p_gen = 0
        if pointer_generation:
            #Query is used here since we only need to 
            #provide an abstractive answer when answering yes or no questions.
            p_gen_input = torch.cat((context, embedded[0], s_t_hat), 1)
            p_gen = self.p_gen_linear(p_gen_input)
            p_gen = torch.sigmoid(p_gen)

        #Standard decoder stuff
        output = torch.cat((embedded[0], context), 1)

        output = self.attn_combine(output).unsqueeze(0)
        output = F.relu(output)

        output = output.view(batch_size, -1)

        output, s_t = self.lstm(output.unsqueeze(0), (h_decoder, c_decoder))
        output = self.out2(self.out1(torch.cat((output[0], context), 1)))
        final_dist = output

        #Pointer generation
        if pointer_generation:
            final_dist = F.softmax(final_dist, dim=1)
            vocab_dist_ = p_gen * final_dist
            attn_dist_ = (1 - p_gen) * attn_weights
            if extra_zeros is not None:
                vocab_dist_ = torch.cat([vocab_dist_, extra_zeros], 1)

            final_dist = vocab_dist_.scatter_add(1, extended, attn_dist_)
        else:
            final_dist = F.log_softmax(final_dist, dim=1)

        coverage_next = coverage

        return final_dist, s_t, attn_weights, coverage_next
    def clear_time(self):
        self.time = int(round(time.time() * 1000))

class AttnSent(nn.Module):
    def __init__(self, hidden_size, dropout_p=0.1, max_length_input=1, max_length_output=1):
        super(AttnSent, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_p = dropout_p
        self.max_length = max_length_input
        self.max_length_out = max_length_output

        self.gcn_reduce = nn.Linear(self.max_length * hidden_size, self.max_length * hidden_size)
        self.attn = nn.Linear(self.max_length * hidden_size, self.max_length_out)
        
        #PG.init_linear_wt(self.attn)
        #PG.init_linear_wt(self.attn_combine)
        
        self.softmax = nn.Softmax(dim=0)
        self.dropout = nn.Dropout(self.dropout_p)
        self.time = int(round(time.time() * 1000))
    
    def forward(self, embedded, device, mask=None):
        embedded_d = embedded
        embedded_d = self.dropout(embedded)
        
        batch_size, _, embd = embedded_d.size()
        embedded_d = embedded_d.view(-1, embd)

        attn_applied = None
        attn = self.attn(torch.tanh(self.gcn_reduce(embedded_d)))
        attn = attn.view(batch_size, -1)

        if mask is not None:
            mask = mask.view(attn.size())
            #assert mask.size() == attn.size()
            mask = mask.eq(0).data

            attn.data.masked_fill_(mask, -float('inf'))
            attn = self.softmax(attn)


        attn_applied = torch.bmm(torch.transpose(embedded, 1, 2), attn.unsqueeze(2)).squeeze(2)
        return attn_applied, attn
        

    def clear_time(self):
        self.time = int(round(time.time() * 1000)) 
    def initHidden(self, device, sentences, batches = 1):
        return torch.zeros(batches, sentences, self.hidden_size, device=device)
#Everything south of here was pulled from graphIE
class GlobalNode(nn.Module):

    def __init__(self, d_input, d_model):
        super(GlobalNode, self).__init__()
        self.d_model = d_model
        self.linear_gi = nn.Linear(d_input, d_model)
        self.linear_go = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        docu_len = mask.float().sum(dim=1)
        mask = mask.unsqueeze(2).expand(-1, -1, self.d_model).float()

        h_g = torch.sum(torch.tanh(self.linear_gi(x)) * mask, dim=1)
        h_g = h_g.div(docu_len.unsqueeze(1).expand_as(h_g))
        
        return self.linear_go(h_g)


class GNN_Layer(nn.Module):

    def __init__(self, d_input, d_model, globalnode, n_graph=1):
        super(GNN_Layer, self).__init__()
        self.linear = nn.Linear(d_input, d_model)
        self.linear_gcn = nn.Linear(d_input*n_graph, d_model)
        self.n_graph = n_graph
        self.globalnode = globalnode
        if globalnode:
            self.g_node = GlobalNode(d_input, d_model)

    def forward(self, x, adjs, mask=None):
        # x: (batch, N, input_dim)
        # adjs: (batch, n_graph, N, N)
        if len(adjs.size()) == 3:
            adjs = adjs.unsqueeze(1)
        batch, num_sent, d_input = x.size()
        assert adjs.size(1) == self.n_graph
        h = self.linear(x)
        x = x.unsqueeze(1).expand(-1, self.n_graph, -1, -1)
        h_gcn = torch.matmul(adjs, x).transpose(1, 2).contiguous().view(batch, num_sent, -1)
        h_gcn = self.linear_gcn(h_gcn)
        d = adjs.sum(dim=3).sum(dim=1).unsqueeze(2)
        d = d + d.eq(0).float()
        h = h + h_gcn / d # batch_size * docu_len * dim
        if self.globalnode:
            h = h + self.g_node(x, mask).unsqueeze(1).expand_as(h)
        h = F.elu(h)
        return h


class GNN_Att_Layer(nn.Module):

    def __init__(self, n_head, d_input, d_model, globalnode, n_graph=1):
        super(GNN_Att_Layer, self).__init__()
        self.linear = nn.Linear(d_input, d_model)
        self.n_head = n_head
        self.attention = MultiHeadAttention(n_head=n_head, d_input=d_input, d_model=d_model)
        self.globalnode = globalnode
        if globalnode:
            self.g_node = GlobalNode(d_input, d_model)

    def forward(self, x, adj, mask=None):
        # x: (N, input_dim)
        graph_mask = adj.ne(0) # .data
        h_gcn, attn = self.attention(x, x, x, graph_mask)
        h = self.linear(x) + h_gcn # batch_size * docu_len * dim
        if self.globalnode:
            h = h + self.g_node(x, mask).unsqueeze(1).expand_as(h)
        h = F.elu(h)
        return h


class GNN_Pos_Att_Layer(nn.Module):

    def __init__(self, d_input, d_model):
        super(GNN_Pos_Att_Layer, self).__init__()
        self.linear1 = nn.Linear(d_input, d_model)
        self.linear2 = nn.Linear(d_model, d_model)
        self.attention = PositionAwareAttention(d_input=d_input, d_model=d_model)

    def forward(self, x, pos, adj):
        # x: (N, input_dim)
        h_gcn, attn = self.attention(x, pos, adj)
        h = self.linear1(x) + self.linear2(h_gcn) # batch_size * docu_len * dim
        h = F.elu(h)
        return h



def graph_masked_mean(x, adj):
    # x: (batch, num_sent, -1)
    # adj: (batch, num_sent, num_sent)
    # return (batch, num_sent, -1)
    mask = adj.ne(0).float()
    length = mask.sum(dim=2)
    y = torch.bmm(mask, x)
    length = length.unsqueeze(2).expand_as(y)
    length = length + length.eq(0).float()
    return y / length


class GNN(nn.Module):

    def __init__(self, word_vocab_size, char_vocab_size, d_output, args):
        super(GNN, self).__init__()
        self.model = args.model

        self.char_cnn = Character_CNN(char_vocab_size, args.d_embed, args.n_filter, args.filter_sizes)
        self.word_emb = nn.Embedding(word_vocab_size, args.d_embed)
        
        self.d_graph = args.d_graph
        self.gnn_layer = nn.ModuleList()
        d_in = args.n_filter * len(args.filter_sizes) + args.d_embed
        
        self.encoder_lstm = Word_LSTM(d_in, args.d_graph[0])

        if self.model in ['lstm-lstm', 'lstm-gcn-lstm', 'lstm-rgcn-lstm', 'lstm-gat-lstm']:
            self.decoder_lstm = Word_LSTM(args.d_graph[0], args.d_graph[-1])
        else:
            self.decoder_lstm = None

        # if self.model == 'concat':
        #     self.concat_transform = nn.Linear(args.d_graph[0]*4, args.d_graph[0])
        #     d_in = args.d_graph[0]
        d_in = args.d_graph[0]

        if self.model in ['lstm-gcn-lstm', 'lstm-rgcn-lstm']:
            self.d_pos_embed = args.d_pos_embed
            self.pos_linear = nn.Linear(4, self.d_pos_embed)
            d_in += self.d_pos_embed
        else:
            self.pos_linear = None
        # if self.model == 'gnnattn':
        #     d_in = args.d_graph[0]
        
        for d_out in args.d_graph:
            # self.lstm_layer.append(Word_LSTM(d_in, d_out))
            if args.model == 'lstm-rgcn-lstm':
                self.gnn_layer.append(GNN_Layer(d_in, d_out, args.globalnode, n_graph=4))
            if args.model == 'lstm-gcn-lstm':
                self.gnn_layer.append(GNN_Layer(d_in, d_out, args.globalnode, n_graph=1))
            if args.model == 'lstm-gat-lstm':
                self.gnn_layer.append(GNN_Pos_Att_Layer(d_in, d_out))  # NOT SUPPORTED
                # self.gnn_layer.append(GNN_Att_Layer(n_head=4, d_input=d_in, d_model=d_out, globalnode=args.globalnode))
            d_in = d_out

        self.out_linear1 = nn.Linear(d_in, d_in)
        self.out_linear2 = nn.Linear(d_in, d_output)

        self.globalnode = args.globalnode
        
        self.drop = nn.Dropout(p=args.dropout)
        # self.log_softmax = nn.LogSoftmax(dim=2)

        self.crf = args.crf
        if self.crf:
            self.crf_layer = CRF(d_output)

        self.clear_time()

    def clear_time(self):
        self.cnn_time, self.lstm_time, self.gcn_time = 0, 0, 0
        # for layer in self.lstm_layer:
        #     layer.lstm.lstm_time = 0

    def print_time(self):
        print('cnn %f  lstm %f  gcn %f'%(self.cnn_time, self.lstm_time, self.gcn_time))

    def forward(self, data, data_word, pos, length, mask, adjs):
        # h = self.clstm(words, length)
        batch_size, docu_len, sent_len, word_len = data.size()

        if self.model == 'lstm-gcn-lstm':
            adjs = adjs.sum(dim=1)

        # word representation
        start = time.time()
        char_emb = self.char_cnn(data.long().view(-1, word_len))
        char_emb = char_emb.view(batch_size*docu_len, sent_len, -1)
        self.cnn_time += time.time()-start

        word_emb = self.word_emb(data_word.view(batch_size*docu_len, -1))
        word_emb = word_emb.view(batch_size * docu_len, sent_len, -1)
        h_word = torch.cat((char_emb, word_emb), dim=2)

        length = length.data.view(batch_size * docu_len)
        mask = mask.view(batch_size * docu_len, sent_len)
        sent_mask = Variable(length.ne(0).view(batch_size, docu_len))
        
        # encoder
        start = time.time()
        h_word = self.drop(h_word)
        h_word, h_sent = self.encoder_lstm(h_word, length, mask, None)
        self.lstm_time += time.time()-start
        
        h_gcn = None
        h_sent = h_sent.view(batch_size, docu_len, -1)
        
        start = time.time()  

        if self.pos_linear:
            feat = F.tanh(self.pos_linear(pos))
            h_gcn = torch.cat((h_sent, feat), dim=2)
        for i in range(len(self.gnn_layer)):
            h_gcn = self.gnn_layer[i](h_gcn, adjs, sent_mask)

        if h_gcn is not None:
            h_gcn = h_gcn.view(batch_size * docu_len, -1)
        
        # if self.model == 'gnnattn': 
        #     h_gcn = h_lstm
        #     for i in range(self.num_layers):
        #         h_gcn = self.gnn_layer[i](h_gcn, pos, adjs.sum(dim=1))            
        #     h_gcn = h_gcn.view(batch_size * docu_len, -1)

        self.gcn_time += time.time()-start
      
        # if self.model == 'lstm+g':
        #     # h_lstm = h_lstm.view(batch_size, docu_len, -1)
        #     h_gcn = graph_masked_mean(h_lstm, adjs[0])
        #     h_gcn = h_gcn.view(batch_size * docu_len, -1)

        # if self.model == 'concat':
        #     # neighbor_mask:  batch_size * 4 * docu_len * docu_len
        #     # h_lstm = h_lstm.view(batch_size, docu_len, -1)
        #     h_lstm = h_lstm.unsqueeze(1).expand(-1, 4, -1, -1).contiguous().view(batch_size*4, docu_len, -1)
        #     neighbor_mask = neighbor_mask.float().view(batch_size*4, docu_len, docu_len)
        #     h_gcn = torch.bmm(neighbor_mask, h_lstm).view(batch_size, 4, docu_len, -1)
        #     h_gcn = h_gcn.transpose(1,2).contiguous().view(batch_size, docu_len, -1)
        #     h_gcn = F.tanh(self.concat_tra nsform(h_gcn))
        #     h_gcn = h_gcn.view(batch_size * docu_len, -1)

        start = time.time()
        if self.decoder_lstm is not None:
            h_word, h_lstm = self.decoder_lstm(h_word, length, mask, h_gcn)
        # elif self.final == 'linear' or self.final == 'attn':
        #     h_gcn = h_gcn.unsqueeze(1).expand(-1, sent_len, -1)
        #     hg = torch.cat([h, h_gcn], dim=2)
        #     if self.final == 'linear': 
        #         h = self.final_layer(hg)
        #     else:
        #         attn_mask = mask.unsqueeze(2).expand(-1, -1, sent_len)
        #         h, _ = self.final_layer(hg, hg, hg, attn_mask)
        #     h = F.relu(h)
        self.lstm_time += time.time()-start

        h = self.drop(h_word)
        h = self.out_linear1(h)
        h = F.relu(h)
        h = self.out_linear2(h)
        
        # output = self.log_softmax(h)
        
        # if self.globalnode:
        #     g = F.log_softmax(self.linear_global(h_g), dim=1)
        #     return output, g
        return h
