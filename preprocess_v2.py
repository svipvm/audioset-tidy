import pandas as pd
import numpy as np
import glob, os, librosa, soundfile, json

import librosa.display
from multiprocessing import Pool

from utils.util_process_ import *

SPLIT_TIME_LENGTH = 5.0 # second

EVAL_MODE = False
global_metadata = None

DATA_DIR = './data/audioset/audios/unbalanced_train_segments'
SAVE_DIR = '/media/ubuntu/HD/Data/Audioset-Seg'

SAVE_LOGMEL_FLAG = False
SAVE_WAV_FLAG = False

params = {
    'train_strong': './metadata/strong/audioset_train_strong.tsv',
    # 'eval_strong': './metadata/strong/audioset_eval_strong.tsv',
    'label_tsv': './metadata/strong/mid_to_display_name.tsv',

    'resampling_rate': 32000,
    'n_fft': 1024,
    'hop_length': 320,
    'win_length': 1024,
    'lower_hertz': 50,
    'upper_hertz': 14000,
    'mel_bins': 64
}

def valid_boundary(end_thre, start_time, end_time):
    if end_thre <= start_time: return False
    time_length = end_time - start_time
    if time_length < 1e-8: return False
    flag1 = ((end_thre - start_time) / time_length) > 0.5
    return flag1

def get_unique_classes(data, axis):
    result = [d[axis] for d in data]
    result = np.unique(np.array(result)).tolist()
    return [class2label[class_name] for class_name in result]

def split_segment_data(wav_file):
    global global_metadata
    global EVAL_MODE

    data_frame = {"segments": [], "wav_id": [], 'start_time': [],
                  'end_time': [], 'classes': []}

    if EVAL_MODE: 
        save_dir = os.path.join(SAVE_DIR, f'data_cut_logmel_{SPLIT_TIME_LENGTH}s_e')
    else:
        save_dir = os.path.join(SAVE_DIR, f'data_cut_logmel_{SPLIT_TIME_LENGTH}s')
    os.makedirs(save_dir, exist_ok=True)
    segment_ids = global_metadata['segment_id'].unique()

    wav_id = os.path.splitext(os.path.basename(wav_file))[0][1:]
    if wav_id not in segment_ids: return

    try:
        wav_data, sr = librosa.load(wav_file, mono=False)
        wav_data = librosa.resample(wav_data, orig_sr=sr, target_sr=params['resampling_rate'])

        wav_meta = global_metadata[global_metadata['segment_id'] == wav_id].copy()
        if wav_meta.shape[0] == 0: return
        patient_dir = wav_file.split('/')[-2].split('_')[-1]

        # num_sample_point = wav_data.shape[0]

        save_png_file = os.path.join(
                save_dir, patient_dir, wav_id, f'{wav_id}.tiff')
        os.makedirs(os.path.dirname(save_png_file), exist_ok=True)
        if SAVE_LOGMEL_FLAG:
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
            save_tiff(save_png_file, logmel)

        pq = PriorityQueue()
        wav_meta = wav_meta.sort_values(by='start_time_seconds')
        max_time = wav_meta['end_time_seconds'].max()
        meta_index, new_wav_id = 0, 0

        start_time_list = wav_meta['start_time_seconds'].tolist()
        end_time_list = wav_meta['end_time_seconds'].tolist()
        class_name_list = wav_meta['class'].tolist()

        for start_time in np.arange(0, max_time, SPLIT_TIME_LENGTH):
            end_time = start_time + SPLIT_TIME_LENGTH
        
            # update: delete invalid data
            while True:
                if pq.is_empty(): break
                min_end_time = pq.top()[0]
                if min_end_time <= start_time:
                    pq.pop()
                else:
                    break

            # update: add valid adata
            while True:
                if meta_index < len(start_time_list) and valid_boundary(end_time, 
                    start_time_list[meta_index], end_time_list[meta_index]): 
                    pq.push(class_name_list[meta_index], end_time_list[meta_index])
                    meta_index += 1
                else:
                    break
                
            class_name = get_unique_classes(pq._queue, axis=-1)
            if len(class_name) == 0: continue

            start_sample_point = int(start_time * params['resampling_rate'])
            end_sample_point = int(end_time * params['resampling_rate'])
            sub_wav = wav_data[start_sample_point:end_sample_point]

            num_smaple_point_ = sub_wav.shape[0]
            num_block_point_  = SPLIT_TIME_LENGTH * params['resampling_rate']
            if num_smaple_point_ < num_block_point_:
                sub_wav = np.pad(sub_wav, (0, int(num_block_point_ - num_smaple_point_)))

            save_wav_file = os.path.join(
                save_dir, patient_dir, wav_id, f'{wav_id}_{new_wav_id}.wav')
            # os.makedirs(os.path.dirname(save_wav_file), exist_ok=True)
            if SAVE_WAV_FLAG:
                soundfile.write(save_wav_file, sub_wav, params['resampling_rate'])

            sub_melspec = spectrogram(data=sub_wav,
                            n_fft=params['n_fft'], 
                            hop_length=params['hop_length'], 
                            win_length=params['win_length'],
                            window='hann',
                            center=True,
                            pad_mode='reflect')
            sub_logmel = logmel_spectrogram(data=sub_melspec,
                                        sr=params['resampling_rate'],
                                        n_fft=params['n_fft'], 
                                        n_mels=params['mel_bins'],
                                        fmin=params['lower_hertz'],
                                        fmax=params['upper_hertz'])
            save_png_file = os.path.join(
                save_dir, patient_dir, wav_id, f'{wav_id}_{new_wav_id}.tiff')
            save_tiff(save_png_file, sub_logmel)

            data_frame['segments'].append(patient_dir)
            data_frame['wav_id'].append(f'{wav_id}_{new_wav_id}')
            data_frame['start_time'].append(start_time)
            data_frame['end_time'].append(end_time)
            data_frame['classes'].append(class_name)
            new_wav_id += 1
            # print(start_time, end_time, class_name)
    except Exception as e:
        print(wav_file, e)

    return pd.DataFrame(data_frame)

def process_seg_wav(base_dir, metadata):
    global global_metadata
    global_metadata = metadata

    wav_dirs = glob.glob(os.path.join(base_dir, '*'))
    if wav_dirs[0][-3:] == 'wav':
        wav_files = wav_dirs
        with Pool(16) as pool:
            pool.map(split_segment_data, wav_files)
    else:
        final_dataframe = []
        for wav_dir in tqdm(wav_dirs, desc='Dirs'):
            wav_files = glob.glob(f'{wav_dir}/*.wav')
            with Pool(16) as pool:
                result_dataframes = pool.map(split_segment_data, wav_files)
            final_dataframe += result_dataframes
        final_dataframe = pd.concat(final_dataframe, ignore_index=True)
    
    metadata_file = os.path.join(SAVE_DIR, 'metadata', 
                        f'train_strong_{SPLIT_TIME_LENGTH}.csv')
    os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
    final_dataframe.to_csv(metadata_file, index=False)

    print("Finished!")


if __name__ == '__main__':
    label_data = pd.read_csv(params['label_tsv'], delimiter='\t', header=None)
    label_data.rename({0: 'label', 1: 'class'}, axis=1, inplace=True)
    label_data['class'] = label_data['class'].apply(lambda x: x.replace(' ', '-'))

    classes = label_data['class'].unique()
    classes = np.sort(classes)
    class2label = {class_name: class_label for class_label, class_name in enumerate(classes)}

    train_data = pd.read_csv(params['train_strong'], delimiter='\t')
    train_data = pd.merge(train_data, label_data, on='label')
    train_data['segment_id'] = train_data['segment_id'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    print("Start to process!")
    process_seg_wav(DATA_DIR, train_data)
    
    class2label_file = os.path.join(SAVE_DIR, 'metadata', 'class2label.json')
    with open(class2label_file, 'w') as f:
        f.write(json.dumps(class2label))
