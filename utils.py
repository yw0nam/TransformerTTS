import numpy as np
import librosa
import os, copy
from scipy import signal
import torch
from dataclasses import dataclass



def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(10000, 2 * (hid_idx // 2) / d_hid)

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def guided_attention(N, T, g=0.2):
    '''Guided attention. Refer to page 3 on the paper.'''
    W = np.zeros((N, T), dtype=np.float32)
    for n_pos in range(W.shape[0]):
        for t_pos in range(W.shape[1]):
            W[n_pos, t_pos] = 1 - np.exp(-(t_pos / float(T) - n_pos / float(N)) ** 2 / (2 * g * g))
    return W

class preprocess():
    
    def __init__(self, config):
        self.config = config
        
    def _pad_data(self, x, length):
        _pad = 0
        return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)

    def _prepare_data(self, inputs):
        max_len = max((len(x) for x in inputs))
        return np.stack([self._pad_data(x, max_len) for x in inputs])


    def _pad_per_step(self, inputs):
        timesteps = inputs.shape[-1]
        return np.pad(inputs, [[0, 0], [0, 0], [0, self.config.outputs_per_step - (timesteps % self.config.outputs_per_step)]], mode='constant', constant_values=0.0)


    def get_param_size(self, model):
        params = 0
        for p in model.parameters():
            tmp = 1
            for x in p.size():
                tmp *= x
            params += tmp
        return params


    def _pad_mel(self, inputs):
        _pad = 0

        def _pad_one(x, max_len):
            mel_len = x.shape[0]
            return np.pad(x, [[0, max_len - mel_len], [0, 0]], mode='constant', constant_values=_pad)
        
        max_len = max((x.shape[0] for x in inputs))
        return np.stack([_pad_one(x, max_len) for x in inputs])
    
    def get_spectrograms(self, fpath):
        '''Parse the wave file in `fpath` and
        Returns normalized melspectrogram and linear spectrogram.
        Args:
        fpath: A string. The full path of a sound file.
        Returns:
        mel: A 2d array of shape (T, n_mels) and dtype of float32.
        mag: A 2d array of shape (T, 1+n_fft/2) and dtype of float32.
        '''
        # Loading sound file
        y, sr = librosa.load(fpath, sr=self.config.sr)

        # Trimming
        y, _ = librosa.effects.trim(y)

        # Preemphasis
        y = np.append(y[0], y[1:] - self.config.preemphasis * y[:-1])

        # stft
        linear = librosa.stft(y=y,
                            n_fft=self.config.n_fft,
                            hop_length=self.config.hop_length,
                            win_length=self.config.win_length)

        # magnitude spectrogram
        mag = np.abs(linear)  # (1+n_fft//2, T)

        # mel spectrogram
        mel_basis = librosa.filters.mel(
            sr=self.config.sr, n_fft=self.config.n_fft, n_mels=self.config.n_mels)  # (n_mels, 1+n_fft//2)
        mel = np.dot(mel_basis, mag)  # (n_mels, t)

        # to decibel
        mel = 20 * np.log10(np.maximum(1e-5, mel))
        mag = 20 * np.log10(np.maximum(1e-5, mag))

        # normalize
        mel = np.clip((mel - self.config.ref_db + self.config.max_db) /
                    self.config.max_db, 1e-8, 1)
        mag = np.clip((mag - self.config.ref_db + self.config.max_db) /
                    self.config.max_db, 1e-8, 1)

        # Transpose
        mel = mel.T.astype(np.float32)  # (T, n_mels)
        mag = mag.T.astype(np.float32)  # (T, 1+n_fft//2)

        return mel, mag


    def spectrogram2wav(self, mag):
        '''# Generate wave file from linear magnitude spectrogram
        Args:
        mag: A numpy array of (T, 1+n_fft//2)
        Returns:
        wav: A 1-D numpy array.
        '''
        # transpose
        mag = mag.T

        # de-noramlize
        mag = (np.clip(mag, 0, 1) * self.config.max_db) - self.config.max_db + self.config.ref_db

        # to amplitude
        mag = np.power(10.0, mag * 0.05)

        # wav reconstruction
        wav = self.griffin_lim(mag**self.config.power)

        # de-preemphasis
        wav = signal.lfilter([1], [1, -self.config.preemphasis], wav)

        # trim
        wav, _ = librosa.effects.trim(wav)

        return wav.astype(np.float32)


    def griffin_lim(self, spectrogram):
        '''Applies Griffin-Lim's raw.'''
        X_best = copy.deepcopy(spectrogram)
        for i in range(self.config.n_iter):
            X_t = self.invert_spectrogram(X_best)
            est = librosa.stft(X_t, self.config.n_fft, self.config.hop_length,
                            win_length=self.config.win_length)
            phase = est / np.maximum(1e-8, np.abs(est))
            X_best = spectrogram * phase
        X_t = self.invert_spectrogram(X_best)
        y = np.real(X_t)

    def invert_spectrogram(self, spectrogram):
        '''Applies inverse fft.
        Args:
        spectrogram: [1+n_fft//2, t]
        '''
        return librosa.istft(spectrogram, self.config.hop_length, win_length=self.config.win_length, window="hann")