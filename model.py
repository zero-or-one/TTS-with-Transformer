import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from modules import *


class TransformerEncoder(nn.Module):

    def __init__(self, hp):
        super(TransformerEncoder, self).__init__()
        # input embedding stem
        self.prenet = EncoderPrenet(hp.symbols_embedding_dim, hp.num_hidden)
        self.pos_enc = PositionalEncoding(hp.num_hidden)
        self.dropout = nn.Dropout(hp.dropout)
        # transformer
        self.transform = nn.ModuleList([TransformerEncLayer(hp.num_hidden, hp.num_ffn, hp.dropout, \
         hp.num_heads, hp.attn_dropout) for _ in range(hp.num_layers)])
        # decoder head
        self.ln_f = nn.LayerNorm(hp.num_hidden)
        

    def forward(self, x, mask):
        x = self.prenet(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for i in range(len(self.transform)):
            x = self.transform[i](x, mask)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, hp):
        super(TransformerDecoder, self).__init__()
        self.prenet = DecoderPrenet(hp.n_mel_channels, hp.num_hidden, hp.num_hidden)
        self.pos_enc = PositionalEncoding(hp.num_hidden)
        self.dropout = nn.Dropout(hp.dropout)
        self.transform = nn.ModuleList([TransformerDecLayer(hp.num_hidden, hp.num_ffn, hp.dropout, \
         hp.num_heads, hp.attn_dropout) for _ in range(hp.num_layers)])
        self.stop_linear = LinearNorm(hp.num_hidden, 1, w_init='sigmoid')
        self.mel_linear = LinearNorm(hp.num_hidden, hp.n_mel_channels * hp.num_outputs)
        self.post_net = PostNet(hp.num_hidden, hp.n_mel_channels, hp.num_outputs)


    def forward(self, x, enc_o, enc_mask):
        x = self.prenet(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        dot_attn = []
        for i in range(len(self.transform)):
            x, attn = self.transform[i](x, enc_o, enc_mask)
            dot_attn.append(attn)
        stop_token = self.stop_linear(x)
        mel = self.mel_linear(x)
        post_mel = self.PostNet(mel)
        return mel, post_mel, stop_token, dot_attn

        
class TransformerTTS(nn.Module):
    def __init__(self, hp):
        super(TransformerTTS, self).__init__()
        self.encoder = TransformerEncoder(hp)
        self.decoder = TransformerDecoder(hp)
        
    def forward(self, x, y, length_x, max_len=None, teacher_forcing=True):
        if length_x is not None:
            enc_mask = torch.ones(x.shape[0], x.shape[1]).to("cuda")
            for j in range(len(length_x)):
                enc_mask[j, length_x[j]:] -= 1
        else:
            enc_mask = None
        enc_o = self.encoder(x, enc_mask)
        if teacher_forcing:
            dec_output = self.decoder(y[:,:-1], enc_o, enc_mask)
        else:
            trg_len = max_len
            if trg_len is None:
                trg_len = y.size(0)
            for i in range(trg_len-1):
                if i == 0:
                    dec_input = y[:,:1]
                dec_output = self.decoder(dec_input, enc_o, enc_mask)
                dec_input = torch.concat((dec_input, dec_output[:,-1:].argmax(-1)), dim=1)
        return dec_output