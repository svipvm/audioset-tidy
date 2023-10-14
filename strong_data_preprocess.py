import pandas as pd
import numpy as np
import glob, os, librosa, soundfile, json, h5py, logging

import librosa.display
from multiprocessing import Pool

from utils.util_process import *

import matplotlib.pyplot as plt
from IPython.display import Audio

DATA_DIR = '/media/ubuntu/HD_new/download/audioset/audios'
META_DIR = '/home/whq/projects/with-dog-audio/audioset-tidy/metadata'
SAVE_DIR = '/media/ubuntu/ssd2t/AIGroup/Audio-Data/audioset_strong'

PROCESS_MODE = 'train'
assert PROCESS_MODE in ['train', 'eval']

params = {
    'sample_rate': 32000,
    'clip_samples': 32000 * 10,
    'crop_second': 2.0,
    'input_size': (int(32000 * 2.0 / 320) + 1, 64),
    'n_fft': 1024,
    'hop_length': 320,
    'win_length': 1024,
    'lower_hertz': 50,
    'upper_hertz': 14000,
    'mel_bins': 64
}

POSITIVE_LABELS, IGNORE_LABELS = None, None
CLASS2LABEL, MID2LABEL = {}, {}

def get_all_strong_wavs(mode):
    wav_ids = strong_seg_df['wav_id'].unique()
    wav_files = []
    
    for wav_id in tqdm(wav_ids, desc='Index file'):
        if mode == 'train':
            wav_file_ = glob.glob(f"{DATA_DIR}/*{mode}*/{wav_id}.wav")
            if len(wav_file_) != 1:
                wav_file_ = glob.glob(f"{DATA_DIR}/*{mode}*/*/{wav_id}.wav")
            if len(wav_file_) != 1:
                # print("Not found:", f"{DATA_DIR}/**/{wav_id}.wav")
                pass
            else:
                wav_file = wav_file_[0]
                wav_files.append(wav_file)
        elif mode == 'eval':
            wav_file_ = glob.glob(f"{DATA_DIR}/*{mode}*/{wav_id}.wav")
            if len(wav_file_) != 1:
                # print("Not found:", f"{DATA_DIR}/**/{wav_id}.wav")
                pass
            else:
                wav_file = wav_file_[0]
                wav_files.append(wav_file)

    print("Found files:", len(wav_files), '/', len(wav_ids))
    return wav_files

if __name__ == '__main__':
    strong_cls_df = pd.read_csv(f"{META_DIR}/strong/mid_to_display_name.tsv", delimiter='\t')
    weak_cls_df = pd.read_csv(f"{META_DIR}/weak/class_labels_indices.csv")

    if PROCESS_MODE == 'train':
        strong_seg_df = pd.read_csv(f"{META_DIR}/strong/audioset_train_strong.tsv", delimiter='\t')
        weak_seg_dfs = [pd.read_csv(f"{META_DIR}/weak/unbalanced_train_segments.csv", sep=", ", engine='python'),
                        pd.read_csv(f"{META_DIR}/weak/balanced_train_segments.csv", sep=", ", engine='python')]
        weak_seg_df = pd.concat(weak_seg_dfs, axis=0)
    elif PROCESS_MODE == 'test':
        strong_seg_df = pd.read_csv(f"{META_DIR}/strong/audioset_eval_strong.tsv", delimiter='\t')
        weak_seg_df = pd.read_csv(f"{META_DIR}/weak/eval_segments.csv")

    strong_cls_df.sort_values(by='display_name', inplace=True)
    strong_cls_df['label'] = list(range(strong_cls_df.shape[0]))

    classes = strong_cls_df['display_name'].unique()
    for class_name in classes:
        temp_data = strong_cls_df[strong_cls_df['display_name'] == class_name].copy()
        CLASS2LABEL[class_name] = int(temp_data['label'].values[0])
        MID2LABEL[temp_data['mid'].values[0]] = int(temp_data['label'].values[0])
    with open(f'{META_DIR}/tiff_class2label.json', 'w') as f:
        f.write(json.dumps({"class2label": CLASS2LABEL, "mid2label": MID2LABEL}))

    strong_seg_df['wav_id'] = strong_seg_df['segment_id'].apply(lambda x : 'Y' + '_'.join(x.split('_')[:-1]))
    strong_seg_df['label_id'] = strong_seg_df['label'].apply(lambda x :MID2LABEL[x])

    weak_seg_df['wav_id'] = weak_seg_df['YTID'].apply(lambda x : 'Y' + x)
    weak_seg_df['label_id'] = weak_seg_df['positive_labels'].apply(
        lambda x : [MID2LABEL[t] if t in MID2LABEL else -1 for t in x[1:-1].split(',')])
    
    POSITIVE_CLSSSES = ['Bark', 'Bow-wow', 'Yip']
    POSITIVE_LABELS = [CLASS2LABEL[x] for x in POSITIVE_CLSSSES]
    IGNORE_CLASSES   = ['Dog', 'Howl', 'Growling', 'Whimper (dog)']
    IGNORE_LABELS = [CLASS2LABEL[x] for x in IGNORE_CLASSES]


