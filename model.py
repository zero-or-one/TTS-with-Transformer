import torch.nn as nn
import torch
import torch.nn.functional as F
from modules import *
from utils import *



class TransformerEncoder(nn.Module):
    def __init__(self, embedding_size, num_hidden):
        """
        :param embedding_size: dimension of embedding
        :param num_hidden: dimension of hidden
        """
        super(TransformerEncoder, self).__init__()
        #self.alpha = nn.Parameter(torch.ones(1))
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(1024, num_hidden, padding_idx=0),
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.encoder_prenet = EncoderPrenet(embedding_size, num_hidden)
        self.layers = clone_module(Attention(num_hidden), 3)
        self.ffns = clone_module(FFN(num_hidden), 3)

    def forward(self, x, pos):
        if self.training:
            c_mask = pos.ne(0).type(torch.float)
            mask = pos.eq(0).unsqueeze(1).repeat(1, x.size(1), 1)
        else:
            c_mask, mask = None, None
        x = self.encoder_prenet(x)
        pos = self.pos_emb(pos)
        x += pos
        x = self.pos_dropout(x)
        #Ps = list()
        #Ms = list()
        for layer, ffn in zip(self.layers, self.ffns):
            x, _ = layer(x, x, mask=mask, query_mask=c_mask)
            x = ffn(x)
            #Ps.append(torch.cat((layer.key.LinearNorm_layer.weight, layer.value.LinearNorm_layer.weight, layer.query.LinearNorm_layer.weight, layer.final_LinearNorm.LinearNorm_layer.weight), 1))
            #Ms.append(torch.cat((ffn.w_1.conv.weight, ffn.w_2.conv.weight), 1))

        return x, c_mask #, Ps


class TransformerDecoder(nn.Module):
    def __init__(self, hp):
        """
        :param num_hidden: dimension of hidden
        """
        super(TransformerDecoder, self).__init__()
        extra = 128 + 128
        num_hidden = hp.num_hidden
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(2048, num_hidden+extra, padding_idx=0), # 1024
                                                    freeze=True)
        self.pos_dropout = nn.Dropout(p=0.1)
        self.alpha = nn.Parameter(torch.ones(1))
        self.decoder_prenet = DecoderPrenet(hp.n_mel_channels, num_hidden * 2, num_hidden, p=0.2)
        self.norm = LinearNorm(num_hidden, num_hidden)
        self.selfattn_layers = clone_module(Attention(num_hidden, enc=False), 3) # HERE
        self.dotattn_layers = clone_module(Attention(num_hidden, enc=False), 3)
        self.ffns = clone_module(FFN(num_hidden, enc=False), 3)
        self.mel_linear = LinearNorm(num_hidden+extra, hp.n_mel_channels * hp.outputs_per_step)
        self.stop_linear = LinearNorm(num_hidden+extra, 1, w_init='sigmoid')
        self.postconvnet = PostNet(num_hidden, hp.n_mel_channels, hp.outputs_per_step)
        self.speaker_mel = SpeakerModule()
        self.speaker_text = SpeakerModule()


    def forward(self, memory, decoder_input, c_mask, pos):
        batch_size = memory.size(0)
        decoder_len = decoder_input.size(1)  
        speaker_id = 0
        speaker_id = torch.LongTensor([speaker_id] * batch_size).to(memory.device)
        speaker_t = self.speaker_text(speaker_id, batch_size, memory.shape[1])
        memory = torch.cat((memory, speaker_t), dim=2)   
        # get decoder mask with triangular matrix
        if self.training:
            m_mask = pos.ne(0).type(torch.float)
            mask = m_mask.eq(0).unsqueeze(1).repeat(1, decoder_len, 1)
            if next(self.parameters()).is_cuda:
                mask = mask + torch.triu(torch.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1,
                                                                                                 1).byte()
            else:
                mask = mask + torch.triu(torch.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            zero_mask = c_mask.eq(0).unsqueeze(-1).repeat(1, 1, decoder_len)
            zero_mask = zero_mask.transpose(1, 2)
        else:
            if next(self.parameters()).is_cuda:
                mask = torch.triu(torch.ones(decoder_len, decoder_len).cuda(), diagonal=1).repeat(batch_size, 1, 1).byte()
            else:
                mask = torch.triu(torch.ones(decoder_len, decoder_len), diagonal=1).repeat(batch_size, 1, 1).byte()
            mask = mask.gt(0)
            m_mask, zero_mask = None, None

        # Decoder pre-network
        decoder_input = self.decoder_prenet(decoder_input)
        decoder_input = self.norm(decoder_input)
        speaker_m = self.speaker_mel(speaker_id, batch_size, decoder_input.shape[1])
        decoder_input = torch.cat((decoder_input, speaker_m), dim=2)
        pos = self.pos_emb(pos)
        decoder_input = pos * self.alpha + decoder_input
        decoder_input = self.pos_dropout(decoder_input)
        attn_dot_list = list()
        #Ps = list()
        #Ms = list()

        for selfattn, dotattn, ffn in zip(self.selfattn_layers, self.dotattn_layers, self.ffns):
            decoder_input, _ = selfattn(decoder_input, decoder_input, mask=mask, query_mask=m_mask)
            decoder_input, attn_dot = dotattn(memory, decoder_input, mask=zero_mask, query_mask=m_mask)
            decoder_input = ffn(decoder_input)
            # decoder_input_stop = ffn(decoder_input, False)
            attn_dot_list.append(attn_dot)
            #Ps.append(torch.cat((selfattn.key.LinearNorm_layer.weight, selfattn.value.LinearNorm_layer.weight, selfattn.query.LinearNorm_layer.weight), 1))
            #Ms.append(selfattn.final_LinearNorm.LinearNorm_layer.weight)

        mel_out = self.mel_linear(decoder_input)
        postnet_input = mel_out.transpose(1, 2)
        out = self.postconvnet(postnet_input)
        out = postnet_input + out
        out = out.transpose(1, 2)
        stop_tokens = self.stop_linear(decoder_input)
        return mel_out, out, attn_dot_list, stop_tokens#, Ps, Ms


class TransformerTTS(nn.Module):
    def __init__(self, hp):
        super(TransformerTTS, self).__init__()
        self.encoder = TransformerEncoder(hp.symbols_embedding_dim, hp.num_hidden)
        self.decoder = TransformerDecoder(hp)

    def forward(self, text, mel_input, pos_text, pos_mel):
        memory, c_mask = self.encoder.forward(text, pos=pos_text)
        mel_output, postnet_output, attn_dot, stop_preds = self.decoder.forward(memory, mel_input, c_mask,
                                                                                                pos=pos_mel)
        return mel_output, postnet_output, stop_preds, attn_dot

    def infer(self, text,  max_len=None):
        if max_len is None:
            max_len = text.shape[1]*9
        mel_input = torch.zeros([1,1, 80]).cuda()
        pos_text = torch.arange(1, text.size(1)+1).unsqueeze(0)
        pos_text = pos_text.cuda()
        for _ in range(max_len):
            pos_mel = torch.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, _, _ = self.forward(text, mel_input, pos_text, pos_mel)
            mel_input = torch.cat([mel_input, mel_pred[:,-1:,:]], dim=1)
        return postnet_pred