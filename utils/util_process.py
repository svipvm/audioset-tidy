import glob, os, sys, tifffile
from tqdm import tqdm

import pandas as pd
import numpy as np
import librosa, heapq


def load_wav(filename, resampling_rate):
    wav_data, sample_rate = librosa.load(filename, mono=False, sr=resampling_rate)
    return wav_data, sample_rate

def save_tiff(filename, data):
    tifffile.imwrite(filename, data)

def spectrogram(data, n_fft=2048, hop_length=None, win_length=None,
        window='hann', center=True, pad_mode='reflect'):
    stft_data = librosa.core.stft(y=data, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=center, pad_mode=pad_mode)
    spec = np.abs(stft_data) ** 2 
    return spec

def logmel_spectrogram(data, sr=22050, n_fft=2048, n_mels=64, fmin=0.0, fmax=None, 
        is_log=True, ref=1.0, amin=1e-10, top_db=None):
    melW = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels,
            fmin=fmin, fmax=fmax).T
    mel_spectrogram = np.matmul(data.T, melW)

    def power_to_db(mel_spectrogram):
        ref_value = ref
        log_spec = 10.0 * np.log10(np.clip(mel_spectrogram, a_min=amin, a_max=np.inf))
        log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

        if top_db is not None:
            if top_db < 0:
                raise librosa.util.exceptions.ParameterError('top_db must be non-negative')
            log_spec = np.clip(log_spec, a_min=log_spec.max().item() - top_db, a_max=np.inf)
        return log_spec

    if is_log:
        logmel = power_to_db(mel_spectrogram)
    else:
        logmel = mel_spectrogram
    return logmel

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (priority, self._index, item))
        self._index += 1

    def pop(self):
        return heapq.heappop(self._queue)[-1]
    
    def top(self):
        return self._queue[0]

    def is_empty(self):
        return len(self._queue) == 0

def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]
    
def select_middle_portion(arr, desired_length):
    if len(arr) < desired_length:
        raise ValueError("Array length is less than the desired length")

    middle_index = len(arr) // 2
    half_length = desired_length // 2
    start_index = middle_index - half_length
    end_index = start_index + desired_length

    selected_portion = arr[start_index:end_index, :]

    return selected_portion