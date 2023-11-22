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

WAV_DIR_NAME = 'audioset_wav_16k'
LOGMEL_DIR_NAME = 'audioset_logmel_16k'
# WAV_DIR_NAME = 'audioset_wav_32k'
# LOGMEL_DIR_NAME = 'audioset_logmel_32k'

PROCESS_MODE = 'eval'
assert PROCESS_MODE in ['train', 'eval']

# params = {
#     'sample_rate': 32000,
#     'clip_samples': 32000 * 10,
#     'crop_second': 2.0,
#     'croped_pad_second': 0.05,
#     'input_size': (int(32000 * 2.0 / 320) + 1, 64),
#     'n_fft': 1024,
#     'hop_length': 320,
#     'win_length': 1024,
#     'lower_hertz': 50,
#     'upper_hertz': 14000,
#     'mel_bins': 64
# }
params = {
    'sample_rate': 16000,
    'clip_samples': 16000 * 10,
    'crop_second': 2.0,
    'croped_pad_second': 0.05,
    'input_size': (int(16000 * 2.0 / 160) + 1, 64),
    'n_fft': 512,
    'hop_length': 160,
    'win_length': 512,
    'lower_hertz': 50,
    'upper_hertz': 8000,
    'mel_bins': 64
}
POSITIVE_CLSSSES = ['Bark', 'Bow-wow', 'Yip']
POSITIVE_LABELS = None # [class2label[x] for x in POSITIVE_CLSSSES]
IGNORE_CLASSES = ['Dog', 'Howl', 'Growling', 'Whimper (dog)']
IGNORE_LABELS = None # [class2label[x] for x in IGNORE_CLASSES]

POSITIVE_LABELS, IGNORE_LABELS = None, None
CLASS2LABEL, MID2LABEL = {}, {}

strong_seg_df, weak_seg_df = None, None
    
def get_all_strong_wavs(mode):
    wav_ids = strong_seg_df['wav_id'].unique()
    wav_files = []
    for wav_id in tqdm(wav_ids, desc='Index file'):
        wav_file_ = glob.glob(f"{DATA_DIR}/*{mode}*/{wav_id}.wav")
        if len(wav_file_) != 1:
            wav_file_ = glob.glob(f"{DATA_DIR}/*{mode}*/*/{wav_id}.wav")
        if len(wav_file_) != 1:
            # print("Not found:", f"{DATA_DIR}/**/{wav_id}.wav")
            pass
        else:
            wav_file = wav_file_[0]
            wav_files.append(wav_file)

    print("Found files:", len(wav_files), '/', len(wav_ids))
    return wav_files

def valid_boundary(end_thre, start_time, end_time):
    if end_thre <= start_time: return False
    time_length = end_time - start_time
    if time_length < 1e-8: return False
    flag1 = ((end_thre - start_time) / time_length) > 0.5
    return flag1

def get_unique_classes(data, axis):
    result = [d[axis] for d in data]
    result = np.unique(np.array(result)).tolist()
    return result

def convert_to_seg(audio_file):
    save_dir = f"{SAVE_DIR}/{WAV_DIR_NAME}/{PROCESS_MODE}"
    os.makedirs(save_dir, exist_ok=True)
    
    wav_id = os.path.splitext(os.path.basename(audio_file))[0]
    weak_temp_data = weak_seg_df[weak_seg_df['wav_id'] == wav_id]
    strong_temp_data = strong_seg_df[strong_seg_df['wav_id'] == wav_id]
    if weak_temp_data.shape[0] == 0 or strong_temp_data.shape[0] == 0:
        return

    weak_include_pos_flag = False
    weak_include_ign_flag = False
    for i, weak_item in weak_temp_data.iterrows():
        if np.isin(weak_item['label_id'], POSITIVE_LABELS).any():
            weak_include_pos_flag = True
        if np.isin(weak_item['label_id'], IGNORE_LABELS).any():
            weak_include_ign_flag = True

    strong_include_pos_flag = False
    strong_include_ign_flag = False
    for i, strong_item in strong_temp_data.iterrows():
        if strong_item['label_id'] in POSITIVE_LABELS:
            strong_include_pos_flag = True
        if strong_item['label_id'] in IGNORE_LABELS:
            strong_include_ign_flag = True
            
    # 不存在正例强标签，并且弱标签包含忽略标签
    if (not strong_include_pos_flag) and weak_include_ign_flag: return

    data_frame = {"wav_id": [], "seg_id": [], 'start_time': [],
                  'end_time': [], 'labels': []}

    (audio, _) = librosa.core.load(audio_file, sr=params['sample_rate'], mono=True)
    # audio = pad_or_truncate(audio, params['clip_samples'])

    save_wav_dir = os.path.join(save_dir, wav_id)
    os.makedirs(save_wav_dir, exist_ok=True)

    pq = PriorityQueue()
    wav_meta = strong_temp_data.sort_values(by='start_time_seconds')
    max_time = wav_meta['end_time_seconds'].max()
    meta_index, new_wav_id = 0, 0

    start_time_list = wav_meta['start_time_seconds'].tolist()
    end_time_list = wav_meta['end_time_seconds'].tolist()
    class_label_list = wav_meta['label_id'].tolist()
    
    for start_time in np.arange(0, max_time, params['crop_second']):
        end_time = start_time + params['crop_second']
    
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
                pq.push(class_label_list[meta_index], end_time_list[meta_index])
                meta_index += 1
            else:
                break

        class_labels = get_unique_classes(pq._queue, axis=-1)
        if len(class_labels) == 0: continue
        if np.isin(class_labels, POSITIVE_LABELS).any() \
            ^ weak_include_pos_flag: continue

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
        if num_smaple_point_ < params['sample_rate'] // 2: continue

        num_block_point_  = (params['crop_second'] + 0.1) * params['sample_rate']
        if num_smaple_point_ < num_block_point_:
            sub_wav = np.pad(sub_wav, (0, int(num_block_point_ - num_smaple_point_)))

        save_wav_file = os.path.join(save_dir, wav_id, f'{wav_id}_{new_wav_id}.wav')
        os.makedirs(os.path.dirname(save_wav_file), exist_ok=True)
        soundfile.write(save_wav_file, sub_wav, params['sample_rate'])

        data_frame['wav_id'].append(wav_id)
        data_frame['seg_id'].append(f'{wav_id}_{new_wav_id}')
        data_frame['start_time'].append(start_time)
        data_frame['end_time'].append(end_time)
        data_frame['labels'].append(class_labels)
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
            
            audio_path = os.path.join(save_dir, wav_id, f'{seg_id}.wav')
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
    strong_cls_df = pd.read_csv(f"{META_DIR}/strong/mid_to_display_name.tsv", delimiter='\t')
    weak_cls_df = pd.read_csv(f"{META_DIR}/weak/class_labels_indices.csv")

    if PROCESS_MODE == 'train':
        strong_seg_df = pd.read_csv(f"{META_DIR}/strong/audioset_train_strong.tsv", delimiter='\t')
        weak_seg_dfs = [pd.read_csv(f"{META_DIR}/weak/unbalanced_train_segments.csv", sep=", ", engine='python'),
                        pd.read_csv(f"{META_DIR}/weak/balanced_train_segments.csv", sep=", ", engine='python')]
        weak_seg_df = pd.concat(weak_seg_dfs, axis=0)
    elif PROCESS_MODE == 'eval':
        strong_seg_df = pd.read_csv(f"{META_DIR}/strong/audioset_eval_strong.tsv", delimiter='\t')
        weak_seg_df = pd.read_csv(f"{META_DIR}/weak/eval_segments.csv", sep=", ", engine='python')

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
    
    POSITIVE_LABELS = [CLASS2LABEL[x] for x in POSITIVE_CLSSSES]
    IGNORE_LABELS = [CLASS2LABEL[x] for x in IGNORE_CLASSES]

    wav_list_file = f"{META_DIR}/audioset_wavs_{PROCESS_MODE}"
    if os.path.isfile(wav_list_file):
        with open(wav_list_file, 'r') as f:
            wav_files = eval(f.read())
    else:
        wav_files = get_all_strong_wavs(PROCESS_MODE)
        with open(wav_list_file, 'w') as f:
            f.write(str(wav_files))

    save_csv_file = f"{SAVE_DIR}/{WAV_DIR_NAME}/{PROCESS_MODE}.csv"
    final_dataframe = []
    with Pool(16) as pool:
        result_dataframes = pool.map(convert_to_seg, wav_files)
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