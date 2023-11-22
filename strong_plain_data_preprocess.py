import pandas as pd
import numpy as np
import glob, os, librosa, soundfile, json, h5py, logging

import librosa.display
from multiprocessing import Pool

from utils.util_process import *

import matplotlib.pyplot as plt
from IPython.display import Audio

DATA_DIR = '/media/ubuntu/HD/Data/audio/clean_dog_voice_dataset/clean_dog_voice_2s_v2'
META_DIR = '/home/whq/projects/with-dog-audio/audioset-tidy/metadata'
SAVE_DIR = '/media/ubuntu/ssd2t/AIGroup/Audio-Data/audioset_strong'

WAV_DIR_NAME = 'plain_wav_32k'
LOGMEL_DIR_NAME = 'plain_logmel_32k'
# WAV_DIR_NAME = 'plain_wav_16k'
# LOGMEL_DIR_NAME = 'plain_logmel_16k'

PROCESS_MODE = 'eval'
assert PROCESS_MODE in ['train', 'eval']

positive_label = None

params = {
    'sample_rate': 32000,
    'crop_second': 2.0,
    'input_size': (int(32000 * 2.0 / 320) + 1, 64),
    'n_fft': 1024,
    'hop_length': 320,
    'win_length': 1024,
    'lower_hertz': 50,
    'upper_hertz': 14000,
    'mel_bins': 64
}
# params = {
#     'sample_rate': 16000,
#     'crop_second': 2.0,
#     'input_size': (int(16000 * 2.0 / 160) + 1, 64),
#     'n_fft': 512,
#     'hop_length': 160,
#     'win_length': 512,
#     'lower_hertz': 25,
#     'upper_hertz': 7000,
#     'mel_bins': 64
# }


def convert_to_seg(audio_file):
    save_dir = f"{SAVE_DIR}/{WAV_DIR_NAME}/{PROCESS_MODE}"
    os.makedirs(save_dir, exist_ok=True)
    
    data_frame = {"wav_id": [], "seg_id": [], 'start_time': [],
                  'end_time': [], 'labels': []}

    slice_file_name = os.path.splitext(os.path.basename(audio_file))[0]
    
    (audio, _) = librosa.core.load(audio_file, sr=params['sample_rate'], mono=True)
    # audio = pad_or_truncate(audio, params['clip_samples'])

    max_time = audio.shape[0] / params['sample_rate']
    meta_index, new_wav_id = 0, 0

    for start_time in np.arange(0, max_time, params['crop_second']):
        end_time = start_time + params['crop_second']

        start_sample_point = int(start_time * params['sample_rate'])
        end_sample_point = int(end_time * params['sample_rate'])
        sub_wav = audio[start_sample_point:end_sample_point]

        num_smaple_point_ = sub_wav.shape[0]
        if num_smaple_point_ < params['crop_second'] * params['sample_rate'] // 2: continue

        num_block_point_  = params['crop_second'] * params['sample_rate']
        if num_smaple_point_ < num_block_point_:
            sub_wav = np.pad(sub_wav, (0, int(num_block_point_ - num_smaple_point_)))

        if np.sum(np.fabs(sub_wav) > 0.1) / params['sample_rate'] < params['crop_second'] * 0.05:
            continue

        save_base_name = f'{slice_file_name}_{new_wav_id}'
        save_wav_file = os.path.join(save_dir, slice_file_name, f"{save_base_name}.wav")
        os.makedirs(os.path.dirname(save_wav_file), exist_ok=True)
        soundfile.write(save_wav_file, sub_wav, params['sample_rate'])

        data_frame['wav_id'].append(slice_file_name)
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
                if isinstance(target, str):
                    hf['target'][n] = np.array(eval(target))
                else:
                    hf['target'][n] = np.array(target)
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
    np.random.seed(42)

    with open(f'{META_DIR}/tiff_class2label.json', 'r') as f:
        class2label = json.loads(f.read())
    positive_label = class2label['class2label']['Bark']
    print('positive_label:', positive_label)

    wav_files = glob.glob(f"{DATA_DIR}/*.wav")
    indexes = np.arange(len(wav_files))
    np.random.shuffle(indexes)
    wav_files = [wav_files[i] for i in indexes]
    train_count = int(len(wav_files) * 0.8)
    if PROCESS_MODE == 'train':
        wav_files = wav_files[:train_count]
    elif PROCESS_MODE == 'eval':
        wav_files = wav_files[train_count:]
    print(PROCESS_MODE, 'count:', len(wav_files))

    final_dataframe = []
    save_csv_file = f"{SAVE_DIR}/{WAV_DIR_NAME}/{PROCESS_MODE}.csv"
    with Pool(16) as pool:
        result_dataframes = pool.map(convert_to_seg, wav_files)
    final_dataframe += result_dataframes
    final_dataframe = pd.concat(final_dataframe, ignore_index=True)
    final_dataframe.to_csv(save_csv_file, index=False)
    
    save_data_file = f"{SAVE_DIR}/{LOGMEL_DIR_NAME}/{PROCESS_MODE}_data.h5"
    convert_to_hdf5(save_data_file, final_dataframe)
    
    save_index_file = f"{SAVE_DIR}/{LOGMEL_DIR_NAME}/{PROCESS_MODE}_index.h5"
    convert_to_indexes(save_index_file, save_data_file)
