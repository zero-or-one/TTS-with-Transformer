import torch
import numpy as np
import argparse
from collections import OrderedDict
import matplotlib.pyplot as plt
import os
from scipy.io.wavfile import write

from hparams import BaseHparams
from text import text_to_sequence
from model import TransformerTTS
from postnet import ModelPostNet
from utils import spectrogram2wav


def load_checkpoint(model, path, single_gpu=True):
    model_dict = torch.load(path)   
    new_model_dict = OrderedDict()
    if not single_gpu:
        model = model.load_state_dict(model_dict)
        return
    else: 
        # remove 'module.' of dataparallel
        for k, v in model_dict['model'].items():
            name = k[7:]
            #print(name)
            new_model_dict[name] = v
        model = model.load_state_dict(new_model_dict)


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_checkpoint', type=str, help='Path to the model checkpoint',
                        default="./checkpoint/transformer.pth.tar" )
    parser.add_argument('--vocoder_checkpoint', type=str, help='Path to the vocoder checkpoint',
                        default="./checkpoint/postnet.pth.tar")
    parser.add_argument('--text', type=str, help='Text to synthesize', default="Copmuter is talking to you.")
    parser.add_argument('--max_len', type=int, help='Maximum nummber of iterations', default=None)
    parser.add_argument('--name', type=str, help='Name to save wav and mel', default="test")
    parser.add_argument('-g', '--visible_gpus', type=str, default="0",
                        required=False, help='CUDA visible GPUs')   
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_gpus
    hp = BaseHparams()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerTTS(hp).to(device)
    vocoder = ModelPostNet().to(device)
    load_checkpoint(model, args.model_checkpoint)
    load_checkpoint(vocoder, args.vocoder_checkpoint)
    model.eval()
    vocoder.eval()
    text = np.asarray(text_to_sequence(args.text, [hp.text_cleaners]))
    text = torch.LongTensor(text).unsqueeze(0)
    text = text.cuda()
    
    with torch.no_grad():
        pred = model.infer(text, args.max_len)
        mag = vocoder.forward(pred)
    plt.imsave(hp.sample_path + f"/{args.name}.png", pred.float().data.cpu().numpy()[0])
    wav = spectrogram2wav(mag.squeeze(0).cpu().numpy())
    write(hp.sample_path + f"/{args.name}.wav", hp.sampling_rate, wav)