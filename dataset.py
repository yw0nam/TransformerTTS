import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import librosa
import numpy as np
from text import text_to_sequence
import collections
from scipy import signal
import torch
import math


class LJDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, config, preprocess):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.
        """
        self.landmarks_frame = pd.read_csv(os.path.join(config.root_dir, config.csv_name), 
                                           sep='|', names=['wav_name', 'text_1', 'text_2'])
        self.config = config
        self.preprocess = preprocess

    def load_wav(self, filename):
        return librosa.load(filename, sr=self.config.sample_rate)

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        wav_name = os.path.join(
            self.config.root_dir, 'wavs', self.landmarks_frame['wav_name'].iloc[idx]) + '.wav'
        text = self.landmarks_frame['text_1'].iloc[idx]

        text = np.asarray(text_to_sequence(
            text, [self.config.cleaners]), dtype=np.int32)
        try:
            mel = np.load(wav_name[:-4] + '.pt.npy')
        except:
            mel, mag = self.preprocess.get_spectrograms(wav_name)
            np.save(wav_name[:-4] + '.pt', mel)
            np.save(wav_name[:-4] + '.mag', mag)
            
        mel_input = np.concatenate(
            [np.zeros([1, self.config.n_mels], np.float32), mel[:-1, :]], axis=0)
        text_length = len(text)
        pos_text = np.arange(1, text_length + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)

        sample = {'text': text, 'mel': mel, 'text_length': text_length,
                  'mel_input': mel_input, 'pos_mel': pos_mel, 'pos_text': pos_text}

        return sample


class Transformer_Collator():

    def __init__(self, preprocess):
        self.preprocess = preprocess
        
    def __call__(self, batch):
        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        text_length = [d['text_length'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        
        text = [i for i, _ in sorted(
            zip(text, text_length), key=lambda x: x[1], reverse=True)]
        mel = [i for i, _ in sorted(
            zip(mel, text_length), key=lambda x: x[1], reverse=True)]
        mel_input = [i for i, _ in sorted(
            zip(mel_input, text_length), key=lambda x: x[1], reverse=True)]
        pos_text = [i for i, _ in sorted(
            zip(pos_text, text_length), key=lambda x: x[1], reverse=True)]
        pos_mel = [i for i, _ in sorted(
            zip(pos_mel, text_length), key=lambda x: x[1], reverse=True)]
        text_length = sorted(text_length, reverse=True)

        text = self.preprocess._prepare_data(text).astype(np.int32)
        mel = self.preprocess._pad_mel(mel)
        mel_input = self.preprocess._pad_mel(mel_input)
        pos_mel = self.preprocess._prepare_data(pos_mel).astype(np.int32)
        pos_text = self.preprocess._prepare_data(pos_text).astype(np.int32)
        
        return_values = {
            'text': torch.LongTensor(text),
            'mel': torch.FloatTensor(mel),
            'mel_input': torch.FloatTensor(mel_input),
            'pos_text' : torch.LongTensor(pos_text),
            'pos_mel' :torch.LongTensor(pos_mel),
            'text_length' : torch.LongTensor(text_length)
        }
        return return_values