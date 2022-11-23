import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from collections import OrderedDict


import hparams as hp
from text.symbols import symbols
from layers import LinearNorm, ConvNorm, clone_layers


class EncoderPrenet(nn.Module):
    """
    Prenet for Encoder. 3 convolutions and projection layers
    """
    def __init__(self, embedding_dim, hidden_dim, dr=0.2):
        """
        -> embedding_dim: projected dimension for each phoneme
        -> hidden_dim: dimension of hidden unit
        -> dr: dropout rate
        """
        super(EncoderPrenet, self).__init__()
        self.embed = nn.Embedding(len(symbols), embedding_dim, padding_idx=0)

        kernel_dim, pad = 5, int(np.floor(5 / 2))
        self.conv1 = ConvNorm(in_channels=embedding_dim,
                          out_channels=hidden_dim,
                          kernel_dim=kernel_dim,
                          padding=pad,
                          w_init='relu')
        self.conv2 = ConvNorm(in_channels=embedding_dim,
                          out_channels=hidden_dim,
                          kernel_dim=kernel_dim,
                          padding=pad,
                          w_init='relu')
        self.conv3 = ConvNorm(in_channels=embedding_dim,
                          out_channels=hidden_dim,
                          kernel_dim=kernel_dim,
                          padding=pad,
                          w_init='relu')
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.relu1 = nn.RelU()
        self.relu2 = nn.RelU()
        self.relu3 = nn.RelU()

        self.drop1 = nn.Dropout(dr)
        self.drop2 = nn.Dropout(dr)
        self.drop3 = nn.Dropout(dr)

        self.proj = LinearNorm(hidden_dim, hidden_dim)

    def forward(self, x):
        x = self.embed(x)
        # embedding dim should be in channels.
        # conv is applied along the text
        x = x.transpose(1, 2) 
        x = self.drop1(self.relu1(self.bn1(self.conv1(x))))
        x = self.drop2(self.relu2(self.bn2(self.conv2(x))))
        x = self.drop3(self.relu3(self.bn3(self.conv3(x))))
        # move it back to apply provections
        x = x.transpose(1, 2) 
        x = self.proj(x) 
        return x


class DecoderPrenet(nn.Module):
    """
    Prenet for Decoder. 2 fully connected layers
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dr=0.5):
        """
        -> input_dim: dimension of input
        -> hidden_dim: dimension of hidden unit
        -> output_dim: dimension of output
        -> dr: dropout rate
        """
        super(DecoderPrenet, self).__init__()
        self.fc1 = LinearNorm(input_dim, hidden_dim)
        self.fc2 = LinearNorm(hidden_dim, output_dim)

        self.relu1 = nn.RelU()
        self.relu2 = nn.RelU()

        self.drop1 = nn.Dropout(dr)
        self.drop2 = nn.Dropout(dr)

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, max_len=4096):
        super().__init__()
        dim = args.nhid_tran
        pos = np.arange(0, max_len)[:, None]
        i = np.arange(0, dim // 2)
        denom = 10000 ** (2 * i / dim)

        pe = np.zeros([max_len, dim])
        pe[:, 0::2] = np.sin(pos / denom)
        pe[:, 1::2] = np.cos(pos / denom)
        pe = torch.from_numpy(pe).float()

        self.register_buffer('pe', pe)

    def forward(self, x):
        # DO NOT MODIFY
        return x + self.pe[:x.shape[1]]


MAX_LEN = 100
class MaskedMultiheadAttention(nn.Module):
    """
    A vanilla multi-head masked attention layer with a projection at the end.
    """
    def __init__(self, mask=False):
        super(MaskedMultiheadAttention, self).__init__()
        assert args.nhid_tran % args.nhead == 0
        # mask : whether to use 
        # key, query, value projections for all heads
        self.key = nn.Linear(args.nhid_tran, args.nhid_tran)
        self.query = nn.Linear(args.nhid_tran, args.nhid_tran)
        self.value = nn.Linear(args.nhid_tran, args.nhid_tran)
        # regularization
        self.attn_drop = nn.Dropout(args.attn_pdrop)
        # output projection
        self.proj = nn.Linear(args.nhid_tran, args.nhid_tran)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        if mask:
            self.register_buffer("mask", torch.tril(torch.ones(MAX_LEN, MAX_LEN)))
        self.nhead = args.nhead
        self.d_k = args.nhid_tran // args.nhead

    def forward(self, q, k, v, mask=None):
        # WRITE YOUR CODE HERE
        Q = self.query(q)
        K = self.key(k)
        V = self.value(v)

        Q = Q.reshape(Q.shape[0], Q.shape[1], self.nhead, -1)
        K = K.reshape(K.shape[0], K.shape[1], self.nhead, -1)
        V = V.reshape(V.shape[0], V.shape[1], self.nhead, -1)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)      
        V = V.transpose(1, 2)  

        S = torch.matmul(Q, K.transpose(3,2))
        S /= self.d_k**(0.5)

        if mask is not None:
            S = S.masked_fill_(mask.unsqueeze(1).unsqueeze(1)==0, np.NINF)  
        try:
            S = S.masked_fill_(self.mask[:S.shape[2], :S.shape[3]]==0, np.NINF) 
        except AttributeError:
            pass

        S = self.attn_drop(torch.softmax(S, dim=-1))
        x = torch.matmul(S, V).transpose(1, 2)
        x = x.reshape(Q.shape[0], x.shape[1], -1)
        x = self.proj(x)

        return x


class TransformerEncLayer(nn.Module):
    def __init__(self):
        super(TransformerEncLayer, self).__init__()
        self.ln1 = nn.LayerNorm(args.nhid_tran)
        self.ln2 = nn.LayerNorm(args.nhid_tran)
        self.attn = MaskedMultiheadAttention()
        self.dropout1 = nn.Dropout(args.resid_pdrop)
        self.dropout2 = nn.Dropout(args.resid_pdrop)
        self.ff = nn.Sequential(
            nn.Linear(args.nhid_tran, args.nff),
            nn.ReLU(), 
            nn.Linear(args.nff, args.nhid_tran)
        )

    def forward(self, x, mask=None):
        # WRITE YOUR CODE HERE
        x = self.ln1(x)
        res = self.attn(x, x, x, mask)
        res = self.dropout1(res)
        x = x + res
        x = self.ln2(x)
        res = self.ff(x)
        res = self.dropout1(res)
        x = x + res
        return x

class TransformerDecLayer(nn.Module):
    def __init__(self):
        super(TransformerDecLayer, self).__init__()
        self.ln1 = nn.LayerNorm(args.nhid_tran)
        self.ln2 = nn.LayerNorm(args.nhid_tran)
        self.ln3 = nn.LayerNorm(args.nhid_tran)
        self.dropout1 = nn.Dropout(args.resid_pdrop)
        self.dropout2 = nn.Dropout(args.resid_pdrop)
        self.dropout3 = nn.Dropout(args.resid_pdrop)
        self.attn1 = MaskedMultiheadAttention(mask=True) # self-attention 
        self.attn2 = MaskedMultiheadAttention() # tgt to src attention
        self.ff = nn.Sequential(
            nn.Linear(args.nhid_tran, args.nff),
            nn.ReLU(), 
            nn.Linear(args.nff, args.nhid_tran)
        )
        
    def forward(self, x, enc_o, enc_mask=None):
        # WRITE YOUR CODE HERE
        x = self.ln1(x)
        res = self.attn1(x, x, x)
        res = self.dropout1(res)
        x = x + res
        x = self.ln2(x)
        res = self.attn2(x, enc_o, enc_o, enc_mask)
        res = self.dropout2(res)
        x = x + res
        x = self.ln3(x)
        res = self.ff(x)      
        res = self.dropout3(res)
        x = x + res
        return x  