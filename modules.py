import torch.nn as nn
import torch
import numpy as np
from collections import OrderedDict


from text.symbols import symbols
from layers import LinearNorm, ConvNorm
from utils import clone_module

class EncoderPrenet(nn.Module):
    def __init__(self, embedding_size, num_hidden):
        """
        :param embedding_size: the suze of text embedding
        :param num_hidden: hidden dimension of encoder
        """
        super(EncoderPrenet, self).__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(len(symbols), embedding_size, padding_idx=0)
        self.conv1 = ConvNorm(embedding_size, num_hidden, 5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')
        self.conv2 = ConvNorm(num_hidden, num_hidden, 5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')

        self.conv3 = ConvNorm(num_hidden, num_hidden, 5,
                          padding=int(np.floor(5 / 2)),
                          w_init='relu')
        self.batch_norm1 = nn.BatchNorm1d(num_hidden)
        self.batch_norm2 = nn.BatchNorm1d(num_hidden)
        self.batch_norm3 = nn.BatchNorm1d(num_hidden)
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        self.projection = LinearNorm(num_hidden, num_hidden)
        self.layer_norm = nn.LayerNorm(num_hidden)

    def forward(self, x):
        x = self.embed(x) 
        x = x.transpose(1, 2) 
        x = self.dropout1(torch.relu(self.batch_norm1(self.conv1(x)))) 
        x = self.dropout2(torch.relu(self.batch_norm2(self.conv2(x)))) 
        x = self.dropout3(torch.relu(self.batch_norm3(self.conv3(x)))) 
        x = x.transpose(1, 2) 
        x = self.layer_norm(self.projection(x)) 
        return x


class DecoderPrenet(nn.Module):
    def __init__(self, xsize, hidden_size, output_size, p=0.5):
        """
        :param xsize: dimension of input
        :param hidden_size: dimension of hidden unit
        :param output_size: dimension of output
        """
        super(DecoderPrenet, self).__init__()
        self.xsize = xsize
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
             ('fc1', LinearNorm(self.xsize, self.hidden_size)),
             ('relu1', nn.ReLU()),
             ('dropout1', nn.Dropout(p)),
             ('fc2', LinearNorm(self.hidden_size, self.output_size)),
             ('relu2', nn.ReLU()),
             ('dropout2', nn.Dropout(p))
        ]))

    def forward(self, x):
        return self.layer(x)


class MultiHead(nn.Module):
    def __init__(self, num_hidden):
        """
        :param num_hidden: dimension of hidden 
        """
        super(MultiHead, self).__init__()
        self.num_hidden = num_hidden
        self.attn_dropout = nn.Dropout(p=0.1)

    def forward(self, key, value, query, mask=None, query_mask=None):
        attn = torch.bmm(query, key.transpose(1, 2))
        attn = attn / (self.num_hidden ** 0.5)
        # make mask
        if mask is not None:
            attn = attn.masked_fill(mask, -2 ** 32 + 1)
            attn = torch.softmax(attn, dim=-1)
        else:
            attn = torch.softmax(attn, dim=-1)
        if query_mask is not None:
            attn = attn * query_mask
        #print('attn', attn.shape)
        #print('val', value.shape)
        result = torch.bmm(attn, value)
        return result, attn


class Attention(nn.Module):
    """
    Attention Network
    """
    def __init__(self, num_hidden, h=4, enc=True):
        """
        :param num_hidden: dimension of hidden
        :param h: num of heads 
        """
        super(Attention, self).__init__()
        self.num_hidden = num_hidden
        self.num_hidden_per_attn = num_hidden // h
        self.h = h
        extra = 0 if enc else 128 + 128
        self.key = LinearNorm(num_hidden+extra, num_hidden, bias=False)
        self.value = LinearNorm(num_hidden+extra, num_hidden, bias=False)
        self.query = LinearNorm(num_hidden+extra, num_hidden, bias=False)
        self.multihead = MultiHead(self.num_hidden_per_attn)
        self.residual_dropout = nn.Dropout(p=0.1)
        self.final_linear = LinearNorm(num_hidden * 2+extra, num_hidden+extra)
        self.layer_norm_1 = nn.LayerNorm(num_hidden+extra)


    def forward(self, memory, decoder_input, mask=None, query_mask=None):
        batch_size = memory.size(0)
        seq_k =  memory.size(1)
        seq_q = decoder_input.size(1)
        # Repeat masks h times
        if query_mask is not None:
            query_mask = query_mask.unsqueeze(-1).repeat(1, 1, seq_k)
            query_mask = query_mask.repeat(self.h, 1, 1)
        if mask is not None:
            mask = mask.repeat(self.h, 1, 1)
        K = self.key(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        V = self.value(memory).view(batch_size, seq_k, self.h, self.num_hidden_per_attn)
        Q = self.query(decoder_input).view(batch_size, seq_q, self.h, self.num_hidden_per_attn)
        K = K.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        V = V.permute(2, 0, 1, 3).contiguous().view(-1, seq_k, self.num_hidden_per_attn)
        Q = Q.permute(2, 0, 1, 3).contiguous().view(-1, seq_q, self.num_hidden_per_attn)
        x, attns = self.multihead(K, V, Q, mask=mask, query_mask=query_mask)
        x = x.view(self.h, batch_size, seq_q, self.num_hidden_per_attn)
        x = x.permute(1, 2, 0, 3).contiguous().view(batch_size, seq_q, -1)
        x = torch.cat([decoder_input, x], dim=-1)
        x = self.final_linear(x)
        x += decoder_input
        x = self.layer_norm_1(x)
        return x, attns


class FFN(nn.Module):
    def __init__(self, num_hidden, enc=True):
        """
        :param num_hidden: dimension of hidden 
        """
        super(FFN, self).__init__()
        extra = 0 if enc else 128 + 128
        self.w_1 = ConvNorm(num_hidden+extra, num_hidden * 4, kernel_size=1, w_init='relu')
        self.w_2 = ConvNorm(num_hidden * 4, num_hidden+extra, kernel_size=1)
        self.dropout = nn.Dropout(p=0.1)
        self.layer_norm = nn.LayerNorm(num_hidden+extra)

    def forward(self, x, stopping=False):
        x0 = x 
        x = x.transpose(1, 2) 
        x = self.w_2(torch.relu(self.w_1(x))) 
        x = x.transpose(1, 2) 
        x1 = x + x0
        x = self.layer_norm(x1)
        if stopping:
            # stopping is done
            return x1
        else:
            return x


class PostNet(nn.Module):
    def __init__(self, num_hidden, n_mel_channels, outputs_per_step):
        """
        :param num_hidden: dimension of hidden 
        """
        super(PostNet, self).__init__()
        self.conv1 = ConvNorm(n_mel_channels * outputs_per_step, num_hidden,
                          kernel_size=5,
                          padding=4,
                          w_init='tanh')
        self.conv2 = ConvNorm(num_hidden, n_mel_channels * outputs_per_step,
                          kernel_size=5,
                          padding=4)
        self.conv_list = clone_module(ConvNorm(num_hidden, num_hidden,
                                     kernel_size=5,
                                     padding=4,
                                     w_init='tanh'), 3)
        self.batch_norm_list = clone_module(nn.BatchNorm1d(num_hidden), 3)
        self.pre_batchnorm = nn.BatchNorm1d(num_hidden)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout_list = nn.ModuleList([nn.Dropout(p=0.1) for _ in range(3)])

    def forward(self, x):
        x = self.dropout1(torch.tanh(self.pre_batchnorm(self.conv1(x)[:, :, :-4])))
        for batch_norm, conv, dropout in zip(self.batch_norm_list, self.conv_list, self.dropout_list):
            x = dropout(torch.tanh(batch_norm(conv(x)[:, :, :-4])))
        x = self.conv2(x)[:, :, :-4]
        return x

# This is for our future experiments
class SpeakerModule(nn.Module):
    def __init__(self, speaker_num=110, embedding_dim=256):
        super(SpeakerModule, self).__init__()
        self.embedding_dim = embedding_dim
        self.embedding = nn.Embedding(speaker_num, embedding_dim)
        self.activation = nn.Softsign()
        self.relu = nn.ReLU()
        self.fc = LinearNorm(embedding_dim, embedding_dim)
      
    def forward(self, speaker_id, batch_size, time):
        out = self.embedding(speaker_id)
        out = self.relu(out)
        out = self.fc(out)
        out = out.unsqueeze(1).repeat(1, time, 1)
        out = self.activation(out)
        return out