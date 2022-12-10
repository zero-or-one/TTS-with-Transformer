import torch
import numpy as np
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import os

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


def infer(model, text, max_len=1000):
    text = torch.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    mel_input = torch.zeros([1,1, 80]).cuda()
    pos_text = torch.arange(1, text.size(1)+1).unsqueeze(0)
    pos_text = pos_text.cuda()
    
    with torch.no_grad():
        _, pred, _, _ = model.forward(text, mel_input, \
         [len(text)], max_len, False)

        #mag = vocoder.forward(pred)
    plt.imsave("samples.png", pred.float().data.cpu().numpy()[0][::-1])
    #wav = spectrogram2wav(mag.squeeze(0).cpu().numpy())
    #write(hp.sample_path + "/test.wav", hp.sr, wav)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, help='Path to the model checkpoint')
    #parser.add_argument('--vocoder_checkpoint', type=int, help='Path to the vocoder checkpoint')
    parser.add_argument('--text', type=str, help='Text to synthesize', default="Hello world")
    parser.add_argument('--max_len', type=int, help='Maximum nummber of iterations', default=400)
    parser.add_argument('-g', '--visible_gpus', type=str, default="5",
                        required=False, help='CUDA visible GPUs')   
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    hp = BaseHparams()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerTTS(hp).to(device)
    #vocoder = ModelPostNet().to(device)
    load_checkpoint(model, args.model_checkpoint)
    model.eval()
    text = np.asarray(text_to_sequence(args.text, [hp.text_cleaners]))
    #vocoder.eval()
    infer(model, text, args.max_len)