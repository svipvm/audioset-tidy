import glob, os, sys, tifffile
from tqdm import tqdm

from multiprocessing import Pool
import pandas as pd
import numpy as np
import librosa

params = {
    'block_second': 1.5,
    'resampling_rate': 32000,
    'n_fft': 1024,
    'hop_length': 750,
    'win_length': 1024,
    'lower_hertz': 50,
    'upper_hertz': 14000,
    'mel_bins': 64
}
DATA_DIR = None
EVAL_MODE = None
SAVE_DIR = None

def load_wav(filename, resampling_rate=params['resampling_rate']):
    wav_data, sample_rate = librosa.load(filename, mono=False, sr=resampling_rate)
    return wav_data, sample_rate

def save_tiff(filename, data):
    tifffile.imwrite(filename, data)

def spectrogram(data, n_fft=2048, hop_length=None, win_length=None,
        window='hann', center=True, pad_mode='reflect'):
    stft_data = librosa.stft(y=data, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=center, pad_mode=pad_mode)
    spectrogram = np.abs(stft_data)
    return spectrogram

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

def convert_to_logmel_data(wav_file):
    global EVAL_MODE
    global SAVE_DIR

    if EVAL_MODE: 
        save_dir = os.path.join(SAVE_DIR, 'data_e_logmel')
    else:
        save_dir = os.path.join(SAVE_DIR, 'data_logmel')
    os.makedirs(save_dir, exist_ok=True)

    seg_class = wav_file.split('/')[-2]
    wav_id = os.path.splitext(os.path.basename(wav_file))[0]

    wav_data, sr = load_wav(wav_file)
    block_point = int(sr * params['block_second'])
    smaple_point = wav_data.shape[0]
    if smaple_point <= block_point:
        wav_data = np.pad(wav_data, (0, block_point - smaple_point))
    else:
        wav_data = wav_data[:block_point]
    
    melspec = spectrogram(data=wav_data,
                        n_fft=params['n_fft'], 
                        hop_length=params['hop_length'], 
                        win_length=params['win_length'],
                        window='hann',
                        center=True,
                        pad_mode='reflect')
    logmel = logmel_spectrogram(data=melspec,
                                sr=params['resampling_rate'],
                                n_fft=params['n_fft'], 
                                n_mels=params['mel_bins'],
                                fmin=params['lower_hertz'],
                                fmax=params['upper_hertz'])

    save_png_file = os.path.join(save_dir, seg_class, f'{wav_id}.tiff')
    os.makedirs(os.path.dirname(save_png_file), exist_ok=True)
    save_tiff(save_png_file, logmel)

def process_seg_logmel(base_dir):
    wav_dirs = glob.glob(os.path.join(base_dir, '*'))
    if wav_dirs[0][-3:] == 'wav':
        wav_files = wav_dirs
        with Pool(16) as pool:
            pool.map(convert_to_logmel_data, wav_files)
    else:
        for wav_dir in tqdm(wav_dirs, desc='Dirs'):
            wav_files = glob.glob(f'{wav_dir}/*.wav')
            with Pool(16) as pool:
                pool.map(convert_to_logmel_data, wav_files)
    print("Finished!")

if __name__ == '__main__':
    DATA_DIR = '/media/ubuntu/HD/Data/Audioset-Seg/data'
    SAVE_DIR = '/media/ubuntu/HD/Data/Audioset-Seg'
    EVAL_MODE = False
    process_seg_logmel(DATA_DIR)

