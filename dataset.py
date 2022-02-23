import pandas as pd
from torch.utils.data import Dataset
import os
import librosa
import numpy as np
from text import text_to_sequence
import torch
from torch.nn.utils.rnn import pad_sequence
import pytorch_lightning as pl
from utils import preprocess
from torch.utils.data import DataLoader

class LJDatasets(Dataset):
    """LJSpeech dataset."""

    def __init__(self, config, preprocess, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the wavs.
        """
        if train:
            self.landmarks_frame = pd.read_csv(os.path.join(config.root_dir, config.train_csv), 
                                            sep='|', names=['wav_name', 'text_1', 'text_2'])
        else:
            self.landmarks_frame = pd.read_csv(os.path.join(config.root_dir, config.val_csv),
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
        pos_text = np.arange(1, len(text) + 1)
        pos_mel = np.arange(1, mel.shape[0] + 1)
        
        stop_token = [0]*(len(mel_input) - 1)
        stop_token += [1]

        sample = {'text': text, 'mel': mel, 'mel_input': mel_input, 
                 'pos_text': pos_text, 'pos_mel': pos_mel, 
                 'stop_token': torch.tensor(stop_token, dtype=torch.float)}

        return sample


class Transformer_Collator():

    def __init__(self, preprocess):
        self.preprocess = preprocess
        
    def __call__(self, batch):
        text = [d['text'] for d in batch]
        mel = [d['mel'] for d in batch]
        mel_input = [d['mel_input'] for d in batch]
        stop_token = [d['stop_token'] for d in batch]
        pos_mel = [d['pos_mel'] for d in batch]
        pos_text = [d['pos_text'] for d in batch]
        text_length = [len(d['text']) for d in batch]
        
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
        stop_token = [i for i, _ in sorted(
            zip(stop_token, text_length), key=lambda x: x[1], reverse=True)]
        
        text = self.preprocess._prepare_data(text).astype(np.int32)
        mel = self.preprocess._pad_mel(mel)
        mel_input = self.preprocess._pad_mel(mel_input)
        pos_mel = self.preprocess._prepare_data(pos_mel).astype(np.int32)
        pos_text = self.preprocess._prepare_data(pos_text).astype(np.int32)
        stop_tokens = pad_sequence(stop_token, batch_first=True, padding_value=0)
        model_input = {
            'text': torch.LongTensor(text),
            'mel_input': torch.FloatTensor(mel_input),
            'pos_text': torch.LongTensor(pos_text),
            'pos_mel': torch.LongTensor(pos_mel),
        }
        label = {
            'mel': torch.FloatTensor(mel),
            'stop_tokens': stop_tokens
        }
        return model_input, label
    
class PartitionPerEpochDataModule(pl.LightningDataModule):

    def __init__(
        self, batch_size, config, num_workers=4
    ):
        super().__init__()
        self.config = config
        self.preprocessor = preprocess(config)
        self.batch_size = batch_size
        self.num_workers = num_workers
        
    def prepare_data(self):
        pass
    def setup(self):
        """
        Anything called here is being distributed across GPUs
        (do many times).  Lightning handles distributed sampling.
        """
        # Build the val dataset
        
        self.val_dataset = LJDatasets(self.config, self.preprocessor, train=False)
        self.train_dataset = LJDatasets(self.config, self.preprocessor, train=True)
        
    def train_dataloader(self):
        """
        This function sends the same file to each GPU and
        loops back after running out of files.
        Lightning will apply distributed sampling to
        the data loader so that each GPU receives
        different samples from the file until exhausted.
        """
        return DataLoader(
            self.train_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Transformer_Collator(self.preprocessor),
            pin_memory=True,
            shuffle=True
        )
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            self.batch_size,
            num_workers=self.num_workers,
            collate_fn=Transformer_Collator(self.preprocessor),
            pin_memory=True,
            shuffle=True
        )
