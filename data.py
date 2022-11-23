
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
import torch

import hparams as hp
from text import text_to_sequence
from utils import load_wav_to_torch


class TTSDataset(Dataset):
    """LJSpeech dataset."""

    def __init__(self, hparams):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.
        """
        self.hparams = hparams
        self.df = pd.read_csv(hparams.csv_path, sep='|', header=None)

    def get_mel(self, filename):
        audio, sampling_rate = librosa.load(filename, sr=self.hparams.sampling_rate)
        audio, _ = librosa.effects.trim(audio)

        audio = np.append(audio[0], audio[1:] - self.hparams.preemphasis * audio[:-1])

        lin_spec = librosa.stft(y=audio,
                            n_fft=self.hparams.n_fft,
                            hop_length=self.hparams.hop_length,
                            win_length=self.hparams.win_length)

        # magnitude spectrogram
        mag = np.abs(lin_spec) 
        # mel spectrogram
        mel_basis = librosa.filters.mel(self.hparams.sampling_rate, self.hparams.n_fft, self.hparams.n_mel_channels)
        mel = np.dot(mel_basis, mag)
        # to decibel
        mel = 20 * np.log10(np.maximum(1e-5, mel))
        mag = 20 * np.log10(np.maximum(1e-5, mag))
        # normalize
        mel = np.clip((mel - self.hparams.ref_db + self.hparams.max_db) / self.hparams.max_db, 1e-8, 1)
        mag = np.clip((mag - self.hparams.ref_db + self.hparams.max_db) / self.hparams.max_db, 1e-8, 1)
        # Transpose
        mel = mel.T.astype(np.float32)
        return mel

    def get_text(self, text):
        text_norm = text_to_sequence(text, [self.hparams.text_cleaners])
        text = np.array(text_norm, dtype=np.int32)
        return text

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = self.get_text(self.df.iloc[idx, 1])
        wav_path = os.path.join(self.hparams.audio_dir, self.df.iloc[idx, 0]) + '.wav'
        mel = self.get_mel(wav_path)
        mel_input = np.concatenate([np.zeros([1, self.hparams.n_mel_channels], np.float32), mel[:-1,:]], axis=0)
        text_length = len(text)
        # get positional information
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {'text': text, 'mel': mel, 'text_length':text_length, 'mel_input':mel_input, 'pos_mel':pos_mel, 'pos_text':pos_text}

        return sample


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """
    def __init__(self):
        pass

    def pad_text(self, x, length):
        _pad = 0
        padded = np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)
        return padded
    
    def pad_mel(self, x, length):
        _pad = 0
        padded = np.pad(x, ((0, length - x.shape[0]), (0, 0)), mode='constant', constant_values=_pad)
        return padded

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram
        PARAMS
        ------
        batch: ['text', 'mel', 'text_length', 'mel_input', 'pos_mel', 'pos_text']
        """
        text = [sample['text'] for sample in batch]
        mel = [sample['mel'] for sample in batch]
        mel_input = [sample['mel_input'] for sample in batch]
        text_length = [sample['text_length'] for sample in batch]
        pos_mel = [sample['pos_mel'] for sample in batch]
        pos_text= [sample['pos_text'] for sample in batch]
        # sort data
        text = [i for i,_ in sorted(zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)
        # PAD data
        max_len = max((len(x) for x in text))
        text = np.stack([self.pad_text(x, max_len) for x in text]).astype(np.int32)
        max_len = max((x.shape[0] for x in mel))
        mel = np.stack([self.pad_mel(x, max_len) for x in mel])
        max_len = max((x.shape[0] for x in mel_input))
        mel_input = np.stack([self.pad_mel(x, max_len) for x in mel_input])
        max_len = max((len(x) for x in pos_mel))
        pos_mel = np.stack([self.pad_text(x, max_len) for x in pos_mel]).astype(np.int32)
        max_len = max((len(x) for x in pos_text))
        pos_text = np.stack([self.pad_text(x, max_len) for x in pos_text]).astype(np.int32)

        collated_sample = torch.LongTensor(text), torch.FloatTensor(mel), torch.FloatTensor(mel_input), torch.LongTensor(pos_text), torch.LongTensor(pos_mel), torch.LongTensor(text_length)
        return collated_sample

def prepare_dataloaders(hparams):
    # Get data loaders
    dataset = TTSDataset(hparams)
    collate_fn = TextMelCollate()
    data_loader = DataLoader(dataset, num_workers=1, shuffle=hparams.shuffle,
                              batch_size=hparams.batch_size, pin_memory=False,
                              drop_last=False, collate_fn=collate_fn)
    return data_loader


# Let's check
if __name__ == '__main__':
    print("Data is loading...")
    dataloader = prepare_dataloaders(hp)
    print("Data is prepared!")
    for text, mel, mel_input, pos_text, pos_mel, text_len in dataloader:
        print('text', text[0])
        print('mel', mel[0])
        break
