import numpy as np
from scipy.io.wavfile import read
import torch


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    print("Seed set to {}".format(seed))

def adjust_learning_rate(optimizer, lr, step_num, warmup_step=4000):
    lr = lr * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    if len(data.shape) == 2:
        data = data.mean(axis=1)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

