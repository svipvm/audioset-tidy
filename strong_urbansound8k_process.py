import pandas as pd
import numpy as np
import glob, os, librosa, soundfile, json, h5py, logging

import librosa.display
from multiprocessing import Pool

from utils.util_process import *

import matplotlib.pyplot as plt
from IPython.display import Audio

DATA_DIR = '/media/ubuntu/HD/Data/UrbanSound8K/UrbanSound8K/audio'
META_DIR = '/home/whq/projects/with-dog-audio/audioset-tidy/metadata'
SAVE_DIR = '/media/ubuntu/ssd2t/AIGroup/Audio-Data/audioset_strong'

WAV_DIR_NAME = 'urbansound8k_wav'
LOGMEL_DIR_NAME = 'urbansound8k_logmel'

PROCESS_MODE = 'eval'
assert PROCESS_MODE in ['train', 'eval']

params = {
    'sample_rate': 32000,
    'clip_samples': 32000 * 10,
    'crop_second': 2.0,
    'croped_pad_second': 0.05,
    'input_size': (int(32000 * 2.0 / 320) + 1, 64),
    'n_fft': 1024,
    'hop_length': 320,
    'win_length': 1024,
    'lower_hertz': 50,
    'upper_hertz': 14000,
    'mel_bins': 64
}

strong_seg_df = None
positive_label = None

def convert_to_seg(wav_fs_id):
    save_dir = f"{SAVE_DIR}/{WAV_DIR_NAME}/{PROCESS_MODE}"
    os.makedirs(save_dir, exist_ok=True)
    
    # wav_id = os.path.splitext(os.path.basename(audio_file))[0]
    # weak_temp_data = weak_seg_df[weak_seg_df['wav_id'] == wav_id]
    strong_temp_data = strong_seg_df[(strong_seg_df['fsID'] == wav_fs_id) &
                                     (strong_seg_df['class'] == 'dog_bark')]
    if strong_temp_data.shape[0] == 0:
        return
    
    data_frame = {"wav_id": [], "seg_id": [], 'start_time': [],
                  'end_time': [], 'labels': []}

    wav_fs_id = str(wav_fs_id)
    save_wav_dir = os.path.join(save_dir, wav_fs_id)
    os.makedirs(save_wav_dir, exist_ok=True)

    for i, item_strong_data in strong_temp_data.iterrows():
        # class_name = item_strong_data['class']
        # if class_name != 'dog_bark': continue
        kfold = item_strong_data['fold']
        slice_file_name = item_strong_data['slice_file_name']
        audio_file = f"{DATA_DIR}/fold{kfold}/{slice_file_name}"

        (audio, _) = librosa.core.load(audio_file, sr=params['sample_rate'], mono=True)
        # audio = pad_or_truncate(audio, params['clip_samples'])

        max_time = audio.shape[0] / params['sample_rate']
        meta_index, new_wav_id = 0, 0

        for start_time in np.arange(0, max_time, params['crop_second']):
            end_time = start_time + params['crop_second']

            if start_time - params['croped_pad_second'] < 0:
                start_time = start_time
                end_time = end_time + params['croped_pad_second'] * 2
            else:
                start_time = start_time - params['croped_pad_second']
                end_time = end_time + params['croped_pad_second']

            start_sample_point = int(start_time * params['sample_rate'])
            end_sample_point = int(end_time * params['sample_rate'])
            sub_wav = audio[start_sample_point:end_sample_point]

            num_smaple_point_ = sub_wav.shape[0]
            if num_smaple_point_ < params['crop_second'] * params['sample_rate'] // 2: continue

            num_block_point_  = (params['crop_second'] + 0.1) * params['sample_rate']
            if num_smaple_point_ < num_block_point_:
                sub_wav = np.pad(sub_wav, (0, int(num_block_point_ - num_smaple_point_)))

            save_base_name = f'{os.path.splitext(slice_file_name)[0]}_{new_wav_id}'
            save_wav_file = os.path.join(save_dir, wav_fs_id, f"{save_base_name}.wav")
            os.makedirs(os.path.dirname(save_wav_file), exist_ok=True)
            soundfile.write(save_wav_file, sub_wav, params['sample_rate'])

            data_frame['wav_id'].append(wav_fs_id)
            data_frame['seg_id'].append(save_base_name)
            data_frame['start_time'].append(start_time)
            data_frame['end_time'].append(end_time)
            data_frame['labels'].append([positive_label])
            new_wav_id += 1
        
    return pd.DataFrame(data_frame)

def convert_to_hdf5(save_hdf5_file, meta_data):
    save_dir = f"{SAVE_DIR}/{WAV_DIR_NAME}/{PROCESS_MODE}"
    os.makedirs(os.path.dirname(save_hdf5_file), exist_ok=True)
    
    audios_num = meta_data.shape[0]
    with h5py.File(save_hdf5_file, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('logmel', shape=((audios_num, *params['input_size'])), dtype=np.float32)
        hf.create_dataset('target', shape=((audios_num, )), dtype=
                                h5py.special_dtype(vlen=np.dtype('int32')))
        hf.attrs.create('sample_rate', data=params['sample_rate'], dtype=np.int32)

        # Pack waveform & target of several audio clips to a single hdf5 file
        # for n, wav_file in enumerate(wav_files):
        for n, meta_item in tqdm(meta_data.iterrows(), desc='Convert to HDF5'):
            wav_id = meta_item['wav_id']
            seg_id = meta_item['seg_id']
            target = meta_item['labels']
            
            audio_path = os.path.join(save_dir, str(wav_id), f'{seg_id}.wav')
            # break
            if os.path.isfile(audio_path):
                # logging.info('{} {}'.format(n, audio_path))
                (audio, _) = librosa.core.load(audio_path, sr=params['sample_rate'], mono=True)
                
                melspec = spectrogram(data=audio,
                                n_fft=params['n_fft'], 
                                hop_length=params['hop_length'], 
                                win_length=params['win_length'],
                                window='hann',
                                center=True,
                                pad_mode='reflect')
                logmel = logmel_spectrogram(data=melspec,
                                            sr=params['sample_rate'],
                                            n_fft=params['n_fft'], 
                                            n_mels=params['mel_bins'],
                                            fmin=params['lower_hertz'],
                                            fmax=params['upper_hertz'])
                logmel = select_middle_portion(logmel, params['input_size'][0])
                audio_name = os.path.basename(audio_path)
                hf['audio_name'][n] = audio_name
                hf['logmel'][n] = logmel
                hf['target'][n] = np.array(eval(target))
            else:
                logging.info('{} File does not exist! {}'.format(n, audio_path))

def convert_to_indexes(save_indexes_file, saved_data_file):
    with h5py.File(saved_data_file, 'r') as hr:
        audios_num = hr['logmel'].shape[0]
        with h5py.File(save_indexes_file, 'w') as hw:
            hw.create_dataset('audio_name', data=hr['audio_name'][:], dtype='S20')
            hw.create_dataset('target', data=hr['target'][:], 
                                dtype=h5py.special_dtype(vlen=np.dtype('int32')))
            hw.create_dataset('hdf5_path', data=[(saved_data_file).encode()
                                                    ] * audios_num, dtype='S200')
            hw.create_dataset('index_in_hdf5', data=np.arange(audios_num), dtype=np.int32)


if __name__ == '__main__':
    strong_seg_df = pd.read_csv(f"{META_DIR}/UrbanSound8K/UrbanSound8K.csv")

    if PROCESS_MODE == 'train':
        train_kfold = list(range(1, 10))
        strong_seg_df = strong_seg_df[strong_seg_df['fold'].isin(train_kfold)]
    elif PROCESS_MODE == 'eval':
        eval_kfold = [10]
        strong_seg_df = strong_seg_df[strong_seg_df['fold'].isin(eval_kfold)]

    print(strong_seg_df.shape)
    print(strong_seg_df.head())

    with open(f'{META_DIR}/tiff_class2label.json', 'r') as f:
        class2label = json.loads(f.read())
    positive_label = class2label['class2label']['Bark']
    print('positive_label:', positive_label)

    wav_fs_ids = strong_seg_df['fsID'].unique()

    final_dataframe = []
    save_csv_file = f"{SAVE_DIR}/{WAV_DIR_NAME}/{PROCESS_MODE}.csv"
    with Pool(16) as pool:
        result_dataframes = pool.map(convert_to_seg, wav_fs_ids)
    final_dataframe += result_dataframes
    final_dataframe = pd.concat(final_dataframe, ignore_index=True)
    final_dataframe.to_csv(save_csv_file, index=False)
    print(final_dataframe.shape)
    final_dataframe.head()
    
    final_dataframe = pd.read_csv(save_csv_file)
    print(final_dataframe.shape)
    final_dataframe.head()

    save_data_file = f"{SAVE_DIR}/{LOGMEL_DIR_NAME}/{PROCESS_MODE}_data.h5"
    convert_to_hdf5(save_data_file, final_dataframe)
    
    save_index_file = f"{SAVE_DIR}/{LOGMEL_DIR_NAME}/{PROCESS_MODE}_index.h5"
    convert_to_indexes(save_index_file, save_data_file)
