import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


from text.symbols import symbols
from layers import LinearNorm, ConvNorm


class EncoderPrenet(nn.Module):
    """
    Prenet for Encoder. 3 convolutions and projection layers
    """
    def __init__(self, embedding_dim, hidden_dim, dr=0.1):
        """
        :param embedding_dim: projected dimension for each phoneme
        :param hidden_dim: dimension of hidden unit
        :param dr: dropout rate
        """
        super(EncoderPrenet, self).__init__()
        self.embed = nn.Embedding(len(symbols), embedding_dim, padding_idx=0)

        kernel_dim, pad = 5, int(np.floor(5 / 2))
        self.conv1 = ConvNorm(in_ch=embedding_dim,
                          out_ch=hidden_dim,
                          kernel_dim=kernel_dim,
                          padding=pad,
                          w_init='relu')
        self.conv2 = ConvNorm(in_ch=embedding_dim,
                          out_ch=hidden_dim,
                          kernel_dim=kernel_dim,
                          padding=pad,
                          w_init='relu')
        self.conv3 = ConvNorm(in_ch=embedding_dim,
                          out_ch=hidden_dim,
                          kernel_dim=kernel_dim,
                          padding=pad,
                          w_init='relu')
        
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()

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
    def __init__(self, input_dim, hidden_dim, output_dim, dr=0.1):
        """
        :param input_dim: dimension of input
        :param hidden_dim: dimension of hidden unit
        :param output_dim: dimension of output
        :param dr: dropout rate
        """
        super(DecoderPrenet, self).__init__()
        self.fc1 = LinearNorm(input_dim, hidden_dim)
        self.fc2 = LinearNorm(hidden_dim, output_dim)

        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.drop1 = nn.Dropout(dr)
        self.drop2 = nn.Dropout(dr)

    def forward(self, x):
        x = self.drop1(self.relu1(self.fc1(x)))
        x = self.drop2(self.relu2(self.fc2(x)))
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, num_hidden, max_len=4096, padding_idx=None, trainable_alpha=None):
        super(PositionalEncoding, self).__init__()
        if trainable_alpha is None:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.0
        dim = num_hidden
        pos = np.arange(0, max_len)[:, None]
        i = np.arange(0, dim // 2)
        denom = 10000 ** (2 * i / dim)

        pe = np.zeros([max_len, dim])
        pe[:, 0::2] = np.sin(pos / denom)
        pe[:, 1::2] = np.cos(pos / denom)
        pe = torch.from_numpy(pe).float()
        # pad 0 for padding_idx
        if padding_idx is not None:
            pe[padding_idx] = 0.
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.alpha*self.pe[:x.shape[1]]


class MaskedMultiheadAttention(nn.Module):
    """
    A vanilla multi-head masked attention layer with a projection at the end.
    """
    def __init__(self, num_hidden, num_head, p, mask=False):
        super(MaskedMultiheadAttention, self).__init__()
        assert num_hidden % num_head == 0
        # mask : whether to use 
        # key, query, value projections for all heads
        self.key = nn.Linear(num_hidden, num_hidden)
        self.query = nn.Linear(num_hidden, num_hidden)
        self.value = nn.Linear(num_hidden, num_hidden)
        # regularization
        self.attn_drop = nn.Dropout(p)
        # output projection
        self.proj = nn.Linear(num_hidden, num_hidden)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        MAX_LEN = 1000
        if mask:
            self.register_buffer("mask", torch.tril(torch.ones(MAX_LEN, MAX_LEN)))
        self.nhead = num_head
        self.d_k = num_hidden // num_head

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

        return x, S


class PostNet(nn.Module):
    """
    Post Convolutional Network (mel --> mel)
    """
    def __init__(self, num_hidden, num_mels, num_outputs):
        """
        :param num_hidden: dimension of hidden 
        """
        super(PostNet, self).__init__()
        self.conv1 = ConvNorm(in_ch=num_mels * num_outputs,
                          out_ch=num_hidden,
                          kernel_dim=5,
                          padding=4,
                          w_init='tanh')                
        self.conv_list = nn.ModuleList([ConvNorm(in_ch=num_hidden, out_ch=num_hidden, kernel_dim=5, padding=4, w_init='tanh') for _ in range(3)])
        self.batch_norm_list = nn.ModuleList([nn.BatchNorm1d(num_hidden) for _ in range(3)])
        
        self.conv2 = ConvNorm(in_ch=num_hidden,
                          out_ch=num_mels * num_outputs,
                          kernel_dim=5,
                          padding=4)

        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)

        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=0.1) for _ in range(3)])

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.dropout1(torch.tanh(self.pre_batchnorm(self.conv1(x)[:, :, :-4])))
        for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
            x = dropout(torch.tanh(batch_norm(conv(x)[:, :, :-4])))
        x = self.conv2(x)[:, :, :-4]
        x = x.transpose(1, 2)
        return x


class TransformerEncLayer(nn.Module):
    def __init__(self, num_hidden, num_ffn, p, num_head, attn_p):
        super(TransformerEncLayer, self).__init__()
        self.ln1 = nn.LayerNorm(num_hidden)
        self.ln2 = nn.LayerNorm(num_hidden)
        self.attn = MaskedMultiheadAttention(num_hidden, num_head, attn_p)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
        self.ff = nn.Sequential(
            nn.Linear(num_hidden, num_ffn),
            nn.ReLU(), 
            nn.Linear(num_ffn, num_hidden)
        )

    def forward(self, x, mask=None):
        res, _ = self.attn(x, x, x, mask)
        res = self.dropout1(res)
        x = x + res
        x = self.ln1(x)
        res = self.ff(x)
        res = self.dropout2(res)
        x = x + res
        x = self.ln2(x)
        return x


class TransformerDecLayer(nn.Module):
    def __init__(self, num_hidden, num_ffn, p, num_head, attn_p):
        super(TransformerDecLayer, self).__init__()
        self.ln1 = nn.LayerNorm(num_hidden)
        self.ln2 = nn.LayerNorm(num_hidden)
        self.ln3 = nn.LayerNorm(num_hidden)
        self.dropout1 = nn.Dropout(p)
        self.dropout2 = nn.Dropout(p)
        self.dropout3 = nn.Dropout(p)
        self.attn1 = MaskedMultiheadAttention(num_hidden, num_head, attn_p, mask=True) # self-attention 
        self.attn2 = MaskedMultiheadAttention(num_hidden, num_head, attn_p, ) # tgt to src attention
        self.ff = nn.Sequential(
            nn.Linear(num_hidden, num_ffn),
            nn.ReLU(), 
            nn.Linear(num_ffn, num_hidden)
        )
        
    def forward(self, x, enc_o, enc_mask=None):
        res, _ = self.attn1(x, x, x)
        res = self.dropout1(res)
        x = x + res
        x = self.ln1(x)
        res, attn = self.attn2(x, enc_o, enc_o, enc_mask)
        res = self.dropout2(res)
        x = x + res
        x = self.ln2(x)
        res = self.ff(x)      
        res = self.dropout3(res)
        x = x + res
        x = self.ln3(x)
        return x, attn