import pandas as pd
import numpy as np
import glob, os, librosa, soundfile 
from multiprocessing import Pool
from tqdm import tqdm
import logging

# DATA_DIR = './data/audioset/audios/unbalanced_train_segments'
DATA_DIR = './data/audioset/audios/eval_segments'
EVAL_MODE = False
RESAMPLE_RATE = 32000
SPLIT_TIME_THR = 3.0
SAVE_DIR = '/media/ubuntu/HD/Data/Audioset-Seg'

PARAMS = {
    'train_strong': './metadata/strong/audioset_train_strong.tsv',
    'eval_strong':  './metadata/strong/audioset_eval_strong.tsv',
    'label_tsv':    './metadata/strong/mid_to_display_name.tsv'
}

global_metadata = None

def split_segment_data(wav_file):
    global global_metadata
    global EVAL_MODE

    if EVAL_MODE: 
        save_dir = os.path.join(SAVE_DIR, 'data_e')
    else:
        save_dir = os.path.join(SAVE_DIR, 'data')
    os.makedirs(save_dir, exist_ok=True)
    segment_ids = global_metadata['segment_id'].unique()

    wav_id = os.path.splitext(os.path.basename(wav_file))[0][1:]
    if wav_id not in segment_ids: return

    try:
        wav_data, sr = librosa.load(wav_file, mono=False)
        wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=RESAMPLE_RATE)

        wav_meta = global_metadata[global_metadata['segment_id'] == wav_id]
        new_wav_id = 0
        for _, meta_item in wav_meta.iterrows():
            start_time_seconds = meta_item['start_time_seconds']
            end_time_seconds = meta_item['end_time_seconds']
            seg_class = meta_item['class']
            if end_time_seconds - start_time_seconds > SPLIT_TIME_THR:
                continue

            start_sample_point = int(start_time_seconds * RESAMPLE_RATE)
            end_sample_point = int(end_time_seconds * RESAMPLE_RATE)
            sub_wav = wav_data[start_sample_point:end_sample_point]
            save_wav_file = os.path.join(save_dir, seg_class, f'{wav_id}_{new_wav_id}.wav')
            os.makedirs(os.path.dirname(save_wav_file), exist_ok=True)
            soundfile.write(save_wav_file, sub_wav, RESAMPLE_RATE)
            new_wav_id += 1
            
    except Exception as e:
        print(wav_file, e)    

def process_seg_wav(base_dir, metadata):
    global global_metadata
    global_metadata = metadata

    wav_dirs = glob.glob(os.path.join(base_dir, '*'))
    if wav_dirs[0][-3:] == 'wav':
        wav_files = wav_dirs
        with Pool(16) as pool:
            pool.map(split_segment_data, wav_files)
    else:
        for wav_dir in tqdm(wav_dirs, desc='Dirs'):
            wav_files = glob.glob(f'{wav_dir}/*.wav')
            with Pool(16) as pool:
                pool.map(split_segment_data, wav_files)
    print("Finished!")

if __name__ == '__main__':

    label_data = pd.read_csv(PARAMS['label_tsv'], delimiter='\t', header=None)
    label_data.rename({0: 'label', 1: 'class'}, axis=1, inplace=True)
    label_data['class'] = label_data['class'].apply(lambda x: x.replace(' ', '-'))

    train_data = pd.read_csv(PARAMS['train_strong'], delimiter='\t')
    train_data = pd.merge(train_data, label_data, on='label')
    train_data['segment_id'] = train_data['segment_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    process_seg_wav(DATA_DIR, train_data)

    # eval_data = pd.read_csv(PARAMS['eval_strong'], delimiter='\t')
    # eval_data = pd.merge(eval_data, label_data, on='label')
    # eval_data['segment_id'] = eval_data['segment_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    # EVAL_MODE = True
    # process_seg_wav(DATA_DIR, eval_data)
