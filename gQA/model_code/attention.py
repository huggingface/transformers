import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import math

class Linear(nn.Module):
    ''' Simple Linear layer with xavier init '''
    def __init__(self, d_in, d_out, bias=True):
        super(Linear, self).__init__()
        self.linear = nn.Linear(d_in, d_out, bias=bias)
        init.xavier_normal(self.linear.weight)

    def forward(self, x):
        return self.linear(x)

class DotProductAttention(nn.Module):
    ''' Dot-Product Attention '''

    def __init__(self, d_model, attn_dropout=0.1):
        super(DotProductAttention, self).__init__()
        self.temper = 1 # np.power(d_model, 0.5)
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, q, k, v, attn_mask=None):
        # q  (mb_size x len_q x d_k)
        # k  (mb_size x len_k x d_k)
        # v  (mb_size x len_v x d_v)
        attn = torch.bmm(q, k.transpose(1, 2)) / self.temper

        if attn_mask is not None:
            assert attn_mask.size() == attn.size(), \
                    'Attention mask shape {} mismatch with Attention logit tensor shape ' \
                    '{}.'.format(attn_mask.size(), attn.size())
            attn.data.masked_fill_(attn_mask, -float('inf'))

        attn = self.softmax(attn)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn


class SingleLayerAttention(nn.Module):

    def __init__(self, d_model, d_k, attn_dropout=0.1):
        super(SingleLayerAttention, self).__init__()
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        # self.linear = nn.Linear(2*d_k, d_k)
        self.weight = nn.Parameter(torch.FloatTensor(d_k, 1))
        self.act = nn.LeakyReLU()

        init.xavier_normal(self.weight)

    def forward(self, q, k, v, attn_mask=None):
        # q  (mb_size x len_q x d_k)
        # k  (mb_size x len_k x d_k)
        # v  (mb_size x len_v x d_v)
        mb_size, len_q, d_k = q.size()
        mb_size, len_k, d_k = k.size()
        q = q.unsqueeze(2).expand(-1, -1, len_k, -1) 
        k = k.unsqueeze(1).expand(-1, len_q, -1, -1)
        # x = torch.cat([q, k], dim=3)
        x = q - k

        # x = self.act(self.linear(x))
        # attn = torch.matmul(x, self.weight).squeeze(3)

        attn = self.act(torch.matmul(x, self.weight).squeeze(3))

        if attn_mask is not None:  # mb_size * len_q * len_k
            assert attn_mask.size() == attn.size()
            attn_mask = attn_mask.eq(0).data
            attn.data.masked_fill_(attn_mask, -float('inf'))

        # attn_mask = attn_mask.float()
        # attn = torch.exp(attn) * attn_mask
        # attn_sum = attn.sum(dim=2)
        # # print(attn_sum)
        # attn_sum = attn_sum + attn_sum.eq(0).float()
        # attn = attn / attn_sum.unsqueeze(2).expand_as(attn)
        attn = self.softmax(attn)
        attn.data.masked_fill_(attn_mask, 0)
        attn = self.dropout(attn)
        output = torch.bmm(attn, v)

        return output, attn



class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_input, d_model, d_input_v=None, dropout=0.1):
        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        d_k, d_v = d_model//n_head, d_model//n_head
        self.d_k = d_k
        self.d_v = d_v

        if d_input_v is None:
            d_input_v = d_input

        self.w_qs = nn.Parameter(torch.FloatTensor(n_head, d_input, d_k))
        self.w_ks = nn.Parameter(torch.FloatTensor(n_head, d_input, d_k))
        self.w_vs = nn.Parameter(torch.FloatTensor(n_head, d_input_v, d_v))

        self.attention = DotProductAttention(d_model)
        # self.attention = SingleLayerAttention(d_model, d_k)
        # self.layer_norm = LayerNormalization(d_model)
        self.proj = Linear(n_head*d_v, d_model)

        self.dropout = nn.Dropout(dropout)

        init.xavier_normal(self.w_qs)
        init.xavier_normal(self.w_ks)
        init.xavier_normal(self.w_vs)

    def forward(self, q, k, v, attn_mask=None):

        d_k, d_v = self.d_k, self.d_v
        n_head = self.n_head

        # residual = q

        mb_size, len_q, d_input = q.size()
        mb_size, len_k, d_input = k.size()
        mb_size, len_v, d_input_v = v.size()

        # treat as a (n_head) size batch
        q_s = q.repeat(n_head, 1, 1).view(n_head, -1, d_input) # n_head x (mb_size*len_q) x d_model
        k_s = k.repeat(n_head, 1, 1).view(n_head, -1, d_input) # n_head x (mb_size*len_k) x d_model
        v_s = v.repeat(n_head, 1, 1).view(n_head, -1, d_input_v) # n_head x (mb_size*len_v) x d_model

        # treat the result as a (n_head * mb_size) size batch
        q_s = torch.bmm(q_s, self.w_qs).view(-1, len_q, d_k)   # (n_head*mb_size) x len_q x d_k
        k_s = torch.bmm(k_s, self.w_ks).view(-1, len_k, d_k)   # (n_head*mb_size) x len_k x d_k
        v_s = torch.bmm(v_s, self.w_vs).view(-1, len_v, d_v)   # (n_head*mb_size) x len_v x d_v

        # perform attention, result size = (n_head * mb_size) x len_q x d_v
        outputs, attns = self.attention(q_s, k_s, v_s, attn_mask=attn_mask.repeat(n_head, 1, 1))

        # back to original mb_size batch, result size = mb_size x len_q x (n_head*d_v)
        outputs = torch.cat(torch.split(outputs, mb_size, dim=0), dim=-1) 

        # project back to residual size
        outputs = self.proj(outputs)
        outputs = self.dropout(outputs)

        # return self.layer_norm(outputs + residual), attns
        return outputs, attns


class PositionAwareAttention(nn.Module):

    def __init__(self, d_input, d_model, dropout=0.1):
        super(PositionAwareAttention, self).__init__()

        self.x_proj = Linear(d_input, d_model)
        self.pos_proj = Linear(4, d_model)
        self.act = nn.Tanh()
        self.weight = nn.Parameter(torch.FloatTensor(d_model, 1))

        self.softmax = nn.Softmax(dim=2)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x, pos, attn_mask=None):

        x = self.act(self.x_proj(x))

        batch_size, num_sent, d_model = x.size()
        x1 = x.unsqueeze(2).expand(-1, -1, num_sent, -1)
        x2 = x.unsqueeze(1).expand(-1, num_sent, -1, -1)

        pos1 = pos.unsqueeze(2).expand(-1, -1, num_sent, -1)
        pos2 = pos.unsqueeze(1).expand(-1, num_sent, -1, -1)

        relative_pos = self.act(self.pos_proj(pos1 - pos2))

        # z = torch.cat([x1, x2, relative_pos], dim=3)
        z = relative_pos
        attn = torch.matmul(z, self.weight).squeeze(3)

        if attn_mask is not None:  # mb_size * len * len
            assert attn_mask.size() == attn.size()
            attn_mask = attn_mask.eq(0).data
            attn.data.masked_fill_(attn_mask, -float('inf'))

        # project back to residual size
        attn = self.softmax(attn)
        attn.data.masked_fill_(attn_mask, 0)
        attn = self.dropout(attn)
        output = torch.bmm(attn, x)

        # return self.layer_norm(outputs + residual), attns
        return output, attn
