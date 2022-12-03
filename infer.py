import torch
import numpy as np
import argparse
from collections import OrderedDict

from model import TransformerTTS
from hparams import BaseHparams
from text import text_to_sequence


def load_checkpoint(model, path, single_gpu=True):
    model_dict = torch.load(path)   
    new_model_dict = OrderedDict()
    if not single_gpu:
        model = model.loaf_state_dict(model_dict)
        return
    else: 
        # remove 'module.' of dataparallel
        for k, v in model_dict.items():
            name = k[7:]
            new_model_dict[name] = v
        model = model.load_state_dict(new_model_dict)


def infer(model, vocoder, text, max_len=1000):
    m.load_state_dict(load_checkpoint(args.restore_step1, "transformer"))
    m_post.load_state_dict(load_checkpoint(args.restore_step2, "postnet"))

    text = np.asarray(text_to_sequence(text, [hp.cleaners]))
    text = torch.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    mel_input = torch.zeros([1,1, 80]).cuda()
    pos_text = torch.arange(1, text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.cuda()
    
    with torch.no_grad():
        for _ in range(args.max_len):
            pos_mel = torch.arange(1,mel_input.size(1)+1).unsqueeze(0).cuda()
            mel_pred, postnet_pred, attn, stop_token, _, attn_dec = model.forward(text, mel_input, pos_text, pos_mel)
            mel_input = t.cat([mel_input, mel_pred[:,-1:,:]], dim=1)

        mag_pred = vocoder.forward(postnet_pred)
        
    wav = spectrogram2wav(mag_pred.squeeze(0).cpu().numpy())
    write(hp.sample_path + "/test.wav", hp.sr, wav)
    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=int, help='Path to the model checkpoint', default=172000)
    parser.add_argument('--text', type=int, help='Text to synthesize', default="Hello world")
    parser.add_argument('--max_len', type=int, help='Maximum nummber of iterations', default=400)
    args = parser.parse_args()

    hp = BaseHparams()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerTTS(hp).to(device)
    vocoder = ModelPostNet().to(device)
    model.eval()
    vocoder.eval()
    infer(model, vocoder, args.text, args.max_len)