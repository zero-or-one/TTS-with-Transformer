import torch.nn as nn
import torch
import torch.nn.functional as F
import math
import numpy as np
from modules import *


class TransformerEncoder(nn.Module):

    def __init__(self):
        super(TransformerEncoder, self).__init__()
        # input embedding stem
        self.tok_emb = nn.Embedding(src_ntoken, args.nhid_tran)
        self.pos_enc = PositionalEncoding()
        self.dropout = nn.Dropout(args.embd_pdrop)
        # transformer
        self.transform = nn.ModuleList([TransformerEncLayer() for _ in range(args.nlayers_transformer)])
        # decoder head
        self.ln_f = nn.LayerNorm(args.nhid_tran)
        

    def forward(self, x, mask):
        # WRITE YOUR CODE HERE
        x = self.tok_emb(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for i in range(len(self.transform)):
            x = self.transform[i](x, mask)
        x = self.ln_f(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self):
        super(TransformerDecoder, self).__init__()
        self.tok_emb = nn.Embedding(trg_ntoken, args.nhid_tran)
        self.pos_enc = PositionalEncoding()
        self.dropout = nn.Dropout(args.embd_pdrop)
        self.transform = nn.ModuleList([TransformerDecLayer() for _ in range(args.nlayers_transformer)])
        self.ln_f = nn.LayerNorm(args.nhid_tran)
        self.lin_out = nn.Linear(args.nhid_tran, trg_ntoken)
        self.lin_out.weight = self.tok_emb.weight


    def forward(self, x, enc_o, enc_mask):
        # WRITE YOUR CODE HERE
        x = self.tok_emb(x)
        x = self.pos_enc(x)
        x = self.dropout(x)
        for i in range(len(self.transform)):
            x = self.transform[i](x, enc_o, enc_mask)
        x = self.ln_f(x)   
        logits = self.lin_out(x)     
        logits /= args.nhid_tran ** 0.5 # Scaling logits. Do not modify this
        return logits

        
class Transformer(nn.Module):
    def __init__(self):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        
    def forward(self, x, y, length_x, max_len=None, teacher_forcing=True):
        # WRITE YOUR CODE HERE
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