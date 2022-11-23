import numpy as np
import random
from scipy.io.wavfile import read
import torch
import pandas as pd
import torch.nn.functional as F
from os import listdir
from os.path import isfile, join, split
from tqdm import tqdm
import librosa




def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    if len(data.shape) == 2:
        data = data.mean(axis=1) # for multichannel audios.
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate