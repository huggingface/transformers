import os
import sys
import torch
import torch.autograd as autograd
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import time

# Depending on arg, build dataset
def get_model(vocab, args):
    print("\nBuilding model...")

    model = RNN(vocab, args)
    if args.cuda:
        return model.cuda()
    else:
        return model



def log_sum_exp(vec, dim=0):
    m, idx = torch.max(vec, dim)
    max_exp = m.unsqueeze(-1).expand_as(vec)
    return m + torch.log(torch.sum(torch.exp(vec - max_exp), dim))



class RNN(nn.Module):

    def __init__(self, d_input, d_hidden=512, n_layers=1, cell_type='LSTM', 
        pooling=None, dropout=0, cuda=True, bidirectional=True):
        
        super(RNN, self).__init__()

        self.d_input = d_input
        self.d_hidden = d_hidden
        self.pooling = pooling
        self.cuda = cuda

        # ------------------------------------------------------------------------
        # Word level RNN
        # ------------------------------------------------------------------------
        self.cell_type = cell_type

        if cell_type == 'GRU':
            self.rnn = nn.GRU(d_input, d_hidden//2, n_layers, bidirectional=bidirectional, batch_first=True)
        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(d_input, d_hidden//2, n_layers, bidirectional=bidirectional, batch_first=True)
        
        # ------------------------------------------------------------------------
        # Fully connected hidden layers
        # ------------------------------------------------------------------------
        # self.fc = nn.Linear(d_hidden, d_output)

        self.dropout = nn.Dropout(dropout)
        self.lstm_time = 0

    def clear_time(self):
        self.lstm_time = 0

    def _sort_tensor(self, input, initial, lengths):
        ''' 
        pack_padded_sequence  requires the length of seq be in descending order to work.
        Returns the sorted tensor, the sorted seq length, and the indices for inverting the order.

        Input:
                input: batch_size, seq_len, *
                lengths: batch_size
        Output:
                sorted_tensor: batch_size-num_zero, seq_len, *
                sorted_len:    batch_size-num_zero
                sorted_order:  batch_size
                num_zero
        '''
        sorted_lengths, sorted_order = lengths.sort(0, descending=True)
        sorted_input = input[sorted_order]
        _, invert_order  = sorted_order.sort(0, descending=False)

        # Calculate the num. of sequences that have len 0
        nonzero_idx = sorted_lengths.nonzero()
        num_nonzero = nonzero_idx.size()[0]
        num_zero = sorted_lengths.size()[0] - num_nonzero

        # temporarily remove seq with len zero
        sorted_input = sorted_input[:num_nonzero]
        sorted_lengths = sorted_lengths[:num_nonzero]

        if initial is not None:
            initial = initial[sorted_order][:num_nonzero]

        return sorted_input, initial, sorted_lengths, invert_order, num_zero

    def _unsort_tensor(self, input, invert_order, num_zero):
        ''' 
        Recover the origin order

        Input:
                input:        batch_size-num_zero, seq_len, hidden_dim
                invert_order: batch_size
                num_zero  
        Output:
                out:   batch_size, seq_len, *
        '''
        if num_zero == 0:
            input = input.index_select(0, autograd.Variable(invert_order))

        else:
            dim0, dim1, dim2 = input.size()
            zero = torch.zeros(num_zero, dim1, dim2)
            zero = autograd.Variable(zero)
            if self.cuda:
                zero = zero.cuda()

            input = torch.cat((input, zero), dim=0)
            # input = input.index_select(0, autograd.Variable(invert_order))
            input = input[invert_order]
        return input

    def _aggregate_last_hidden(self, input, text_len):
        '''
        Retrieve the last hidden state for all examples in the batch.
        input: batch_size * max_text_len * rnn_size
        output: batch_size * rnn_size
        '''
        out = []
        for i, last in enumerate(text_len):
            out.append(input[i:i+1, last-1, :])
        out = torch.cat(out, dim=0)
        return out

    def _aggregate_max_pooling(self, input, text_mask):
        '''
        Apply max pooling over time
        input: batch_size * max_text_len * rnn_size
        output: batch_size * rnn_size
        '''
        # zero out padded tokens
        batch_size, max_text_len, _ = input.size()
        # idxes = torch.arange(0, int(torch.max(text_len)), out=torch.LongTensor(torch.max(text_len))).unsqueeze(0).cuda()
        # text_mask = Variable((idxes < text_len.unsqueeze(1)).unsqueeze(2).float())
        input = input * text_mask.detach().unsqueeze(2).float()

        out, _ = torch.max(input, dim=1)

        return out

    def _aggregate_avg_pooling(self, input, text_mask):
        '''
        Apply mean pooling over time
        input: batch_size * max_text_len * rnn_size
        output: batch_size * rnn_size
        '''
        # zero out padded tokens
        batch_size, max_text_len, _ = input.size()
        #print(text_mask.size())
        # idxes = torch.arange(0, int(torch.max(text_len)), out=torch.LongTensor(torch.max(text_len))).unsqueeze(0).cuda()
        # text_mask = Variable((idxes < text_len.unsqueeze(1)).unsqueeze(2).float())
        input = input * text_mask.detach().unsqueeze(2).float()

        out = torch.sum(input, dim=1)
        text_len = text_mask.float().sum(dim=1)
        text_len = text_len.unsqueeze(1).expand_as(out)
        text_len = text_len + text_len.eq(0).float()
        out = out.div(text_len)
        return out


    def forward(self, text, text_len, text_mask, initial=None):
        """
        Input:
            word:     batch_size, max_doc_len
            doc_len:  batch_size
        """
        batch_size, max_text_len, _ = text.size()

        # ------------------------------------------------------------------------
        # Get the word, POS, NER embeddings and concatenate them together
        # ------------------------------------------------------------------------
        word = self.dropout(text)
        # print(word.size())

        # Go through the rnn
        # Sort the word tensor according to the sentence length, and pack them together
        sort_word, sort_initial, sort_len, invert_order, num_zero = self._sort_tensor(input=word, initial=initial, lengths=text_len)
        word = pack_padded_sequence(sort_word, lengths=sort_len.cpu().numpy(), batch_first=True)

        # Run through the word level RNN
        start = time.time()
        if initial is None:
            word, _ = self.rnn(word)         # batch_size, max_doc_len, args.word_hidden_size
        else:
            initial = torch.transpose(sort_initial, 0,1).contiguous()
            if self.cell_type == 'LSTM':
                initial = (initial, initial)
            word, _ = self.rnn(word, initial)
        self.lstm_time += time.time() - start

        # Unpack the output, and invert the sorting
        word = pad_packed_sequence(word, batch_first=True)[0] # batch_size, max_doc_len, rnn_size
        word = self._unsort_tensor(word, invert_order, num_zero) # batch_size, max_doc_len, rnn_size

        if self.pooling == None:
            return word

        if self.pooling == "last":   # use the last hidden state
            out = self._aggregate_last_hidden(word, text_len)
        elif self.pooling == "max":  # max pooling the rnn hidden state
            out = self._aggregate_max_pooling(word, text_mask)
        elif self.pooling == "mean": # mean pooling
            out = self._aggregate_avg_pooling(word, text_mask)

        # Fully connected hidden ReLU layer
        # out = F.relu(self.fc1(out))   # batch_size, args.hidden_dim
        # out = self.dropout(out)
        # out = self.fc(out)           # batch_size, num_classes
 
        return word, out



class CRF(nn.Module):

    def __init__(self, d_output):
        super(CRF, self).__init__()
        self.d_output = d_output + 2

        self.START_TAG = d_output
        self.STOP_TAG = d_output + 1

        # Maps the output of the LSTM into tag space.
        # self.hidden2tag = nn.Linear(self.d_hidden, self.d_output)

        # Matrix of transition parameters.  Entry i,j is the score of
        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.d_output, self.d_output))


    def _compute_partition(self, logits, x_mask):
        '''
        Compute the partition function by dynamic programming
        logits:  batch, len, d_output
        x_mask:  batch, len
        '''
        # print('logits', logits.size())
        # print('x_mask', x_mask.size())

        batch_size, doc_len, d_output = logits.size()
        init_alphas = torch.ones(batch_size, self.d_output) * (-10000.)  # starting from otherwise has zero score
        init_alphas[:,self.START_TAG] = 0                                       # start tag has all of the score
        cumulate = autograd.Variable(init_alphas).cuda()

        # iterating though the sentence
        for i in range(doc_len):
            cur_tag = logits[:, i, :]                                                                   # batch, 5
            cur_tag_broadcast  = cur_tag.unsqueeze(-1).expand(batch_size, *self.transitions.size())     # batch, 5 (cur),  5 (copy)
            cumulate_broadcast = cumulate.unsqueeze(1).expand(batch_size, *self.transitions.size())     # batch, 5 (copy), 5 (prev)
            transit_broadcast  = self.transitions.unsqueeze(0).expand_as(cumulate_broadcast)            # batch, 5 (cur),  5 (prev)

            mat = transit_broadcast + cumulate_broadcast + cur_tag_broadcast

            cumulate_nxt = log_sum_exp(mat, 2).squeeze(-1)                                              # batch, 5 (cur)

            cur_mask = x_mask[:, i].unsqueeze(-1)                                                       # batch, 1
            cumulate = cur_mask * cumulate_nxt + (1-cur_mask) * cumulate                                # batch, 5 (cur)

        cumulate = cumulate + self.transitions[self.STOP_TAG,:].unsqueeze(0).expand_as(cumulate)

        partition = log_sum_exp(cumulate, 1).squeeze(-1)                                                # batch

        return partition

    # def _forward_alg(self, feats):
    #     # Do the forward algorithm to compute the partition function
    #     init_alphas = torch.full((1, self.tagset_size), -10000.)
    #     # START_TAG has all of the score.
    #     init_alphas[0][self.START_TAG] = 0.

    #     # Wrap in a variable so that we will get automatic backprop
    #     forward_var = init_alphas

    #     # Iterate through the sentence
    #     for feat in feats:
    #         alphas_t = []  # The forward tensors at this timestep
    #         for next_tag in range(self.d_output):
    #             # broadcast the emission score: it is the same regardless of
    #             # the previous tag
    #             emit_score = feat[next_tag].view(1, -1).expand(1, self.d_output)
    #             # the ith entry of trans_score is the score of transitioning to
    #             # next_tag from i
    #             trans_score = self.transitions[next_tag].view(1, -1)
    #             # The ith entry of next_tag_var is the value for the
    #             # edge (i -> next_tag) before we do log-sum-exp
    #             next_tag_var = forward_var + trans_score + emit_score
    #             # The forward variable for this tag is log-sum-exp of all the
    #             # scores.
    #             alphas_t.append(log_sum_exp(next_tag_var).view(1))
    #         forward_var = torch.cat(alphas_t).view(1, -1)
    #     terminal_var = forward_var + self.transitions[self.STOP_TAG]
    #     alpha = log_sum_exp(terminal_var)
    #     return alpha

    def _score_sentence(self, logits, x_mask, x_len, true_tag):
        '''
        Return the log(exp(Score)) = score of give tagging sequence
            logits:   batch, len, d_output
            x_mask:   batch, len
            true_tag: batch, len
        '''

        # calculate emission score
        emission_score = torch.gather(logits, 2, true_tag.unsqueeze(-1)).squeeze(-1)   # batch, len
        emission_score = torch.sum(emission_score * x_mask, 1)                         # batch

        # calculate transition score
        batch_size, doc_len, d_output = logits.size()
        # deal with the start tag
        tran_score = torch.gather(self.transitions[:,self.START_TAG], 0, true_tag[:,0])            # batch

        if doc_len > 1:
            
            trn_expand = self.transitions.unsqueeze(0).expand(batch_size, d_output, d_output)          # batch, 5, 5

            tag_r = true_tag[:,1:]                                                         # batch, len-1
            tag_r = tag_r.unsqueeze(-1).expand(-1, -1, d_output)                           # batch, len-1, 5

            trn_row = torch.gather(trn_expand, 1, tag_r)                                   # batch, len-1, 5(prev)

            tag_l = true_tag[:,:-1].unsqueeze(-1)                                          # batch, len-1, 1
            score = torch.gather(trn_row, 2, tag_l).squeeze(-1)                            # batch, len-1

            score = score * x_mask[:,1:]                                                   # batch, len

            tran_score += score.sum(1)                                                     # batch

        # last_tag = torch.gather(true_tag, 1, F.relu(x_len-1)).squeeze(-1)
        # tran_score += torch.gather(self.transitions[self.STOP_TAG,:], 0, last_tag)         # batch (need to pad true_tag by last tag)
        for i in range(batch_size):
            tran_score[i] += self.transitions[self.STOP_TAG, true_tag[i, x_len[i]-1]]

        return tran_score + emission_score

    # def _score_sentence(self, feats, tags):
    #     # Gives the score of a provided tag sequence
    #     score = torch.zeros(1)
    #     tags = torch.cat([torch.tensor([self.START_TAG], dtype=torch.long), tags])
    #     for i, feat in enumerate(feats):
    #         score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
    #     score = score + self.transitions[self.STOP_TAG, tags[-1]]
    #     return score

    def _pad_2(self, x):
        batch_size, doc_len, d_output = x.size()
        x = torch.cat([x, Variable(torch.ones(batch_size, doc_len, 2)*(-10000.)).cuda()], dim=2)
        return x


    def viterbi_decode(self, x, x_mask):
        '''
        Use viterbi algorithm to decode the best tagging sequence

        Input:
            x:        batch_size, doc_len, d_output
            x_mask:   batch_size, doc_len

        Output: bast tagging sequence
            tags:     batch_size, doc_len
            score:    batch_size
        '''
        x = self._pad_2(x)
        
        batch_size, doc_len, d_output = x.size() 
        x = x.transpose(0, 1)

        vit = torch.ones(batch_size, d_output)* (-10000.)
        vit[:, self.START_TAG] = 0
        vit = Variable(vit).cuda()

        pointers = []
        x_mask = x_mask.float()
        for i, logit in enumerate(x):                                             # batch_size, 5
            vit_exp = vit.unsqueeze(1).expand(batch_size, d_output, d_output)     # batch_size, 5, 5 

            # compute transition probability
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)            # batch_size, 5(cur), 5(prev)

            vit_trn_sum = vit_exp + trn_exp                                       # batch_size, 5(cur), 5(prev)

            vt_max, vt_argmax = vit_trn_sum.max(2)                                # batch, 5(cur) for both

            # save best indices, this is only based on the transition probability
            pointers.append(vt_argmax.unsqueeze(0))                               # [1, batch_size, 5]

            # add emission probability
            vit_nxt = vt_max + logit                                              # batch_size, 5

            mask = x_mask[:,i].unsqueeze(-1)                                      # batch_size, 1

            vit = mask * vit_nxt + (1 - mask) * vit                               # batch_size, 5

        # add the transition probability to the stop tag
        vit += self.transitions[self.STOP_TAG,:].unsqueeze(0)                     # batch_size, 5

        pointers = torch.cat(pointers, dim=0)                                     # doc_len, batch_size, 5
        scores, idx = vit.max(1)                                                  # batch for both

        idx = idx.unsqueeze(1)                                                    # batch, 1
        tags = [idx]                                                              # [batch, 1]

        for i in range(doc_len):
            mask = x_mask[:,-1-i].unsqueeze(-1).long()                            # batch, 1
            idx = (1-mask) * idx + mask * torch.gather(pointers[-1-i,:,:], 1, idx)# batch, 1
            tags.insert(0, idx)

        tags = torch.cat(tags[1:], 1)                                             # batch, doc_len
        scores = scores.squeeze(-1)                                               # batch

        return scores, tags


    # def _viterbi_decode(self, feats):
    #     backpointers = []

    #     # Initialize the viterbi variables in log space
    #     init_vvars = torch.full((1, self.d_output), -10000.)
    #     init_vvars[0][self.START_TAG] = 0

    #     # forward_var at step i holds the viterbi variables for step i-1
    #     forward_var = init_vvars
    #     for feat in feats:
    #         bptrs_t = []  # holds the backpointers for this step
    #         viterbivars_t = []  # holds the viterbi variables for this step

    #         for next_tag in range(self.d_output):
    #             # next_tag_var[i] holds the viterbi variable for tag i at the
    #             # previous step, plus the score of transitioning
    #             # from tag i to next_tag.
    #             # We don't include the emission scores here because the max
    #             # does not depend on them (we add them in below)
    #             next_tag_var = forward_var + self.transitions[next_tag]
    #             best_tag_id = argmax(next_tag_var)
    #             bptrs_t.append(best_tag_id)
    #             viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
    #         # Now add in the emission scores, and assign forward_var to the set
    #         # of viterbi variables we just computed
    #         forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
    #         backpointers.append(bptrs_t)

    #     # Transition to STOP_TAG
    #     terminal_var = forward_var + self.transitions[self.STOP_TAG]
    #     best_tag_id = argmax(terminal_var)
    #     path_score = terminal_var[0][best_tag_id]

    #     # Follow the back pointers to decode the best path.
    #     best_path = [best_tag_id]
    #     for bptrs_t in reversed(backpointers):
    #         best_tag_id = bptrs_t[best_tag_id]
    #         best_path.append(best_tag_id)
    #     # Pop off the start tag (we dont want to return that to the caller)
    #     start = best_path.pop()
    #     assert start == self.START_TAG  # Sanity check
    #     best_path.reverse()
    #     return path_score, best_path

    def loglikelihood(self, x, x_mask, x_len, true_tag):
        '''
        Compute the log likelihood of the true tag. 

        Input:
            x:        batch_size, doc_len, d_output
            x_mask:   batch_size, doc_len
            true_tag: batch_size, doc_len (copy the last tag to the last column)

        Output: log likelihood of a given true tag sequence
            score:    batch_size
        '''

        logits = self._pad_2(x)

        x_mask = x_mask.float()          # batch_size, doc_len

        partition = self._compute_partition(logits, x_mask)     # batch

        score = self._score_sentence(logits, x_mask, x_len, true_tag)  # batch

        return (score - partition)

    # def neg_log_likelihood(self, feats, tags):
    #     # feats = self._get_lstm_features(sentence)
    #     forward_score = self._forward_alg(feats)
    #     gold_score = self._score_sentence(feats, tags)
    #     return forward_score - gold_score

    # def forward(self, lstm_feats):  # dont confuse this with _forward_alg above.
    #     # Get the emission scores from the BiLSTM
    #     # lstm_feats = self._get_lstm_features(sentence)

    #     # Find the best path, given the features.
    #     score, tag_seq = self._viterbi_decode(lstm_feats)
    #     return score, tag_seq

