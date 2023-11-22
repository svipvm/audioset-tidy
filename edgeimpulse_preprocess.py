import glob, json, os, librosa, soundfile
from tqdm import tqdm
import numpy as np
import pandas as pd
from xml.dom.minidom import parse as XMLParse

from multiprocessing import Pool
from utils.util_process import *

META_DIR = '/home/whq/projects/with-dog-audio/audioset-tidy/metadata'
SAVE_DIR = '/media/ubuntu/ssd2t/AIGroup/Audio-Data/ei-audios-data'

PROCESS_MODE = 'eval'
WAV_DIR_NAME = '16khz-2s'
PARAMS = {
    'sample_rate': 16000,
    'crop_second': 2.0
}

IGNORE_LABELS = None
POSITIVE_LABELS = None
strong_seg_df = None
weak_seg_df = None

def audioset_get_all_strong_wavs(data_dir, strong_seg_df, mode):
    wav_ids = strong_seg_df['wav_id'].unique()
    wav_files = []
    for wav_id in tqdm(wav_ids, desc='Index file'):
        wav_file_ = glob.glob(f"{data_dir}/*{mode}*/{wav_id}.wav")
        if len(wav_file_) != 1:
            wav_file_ = glob.glob(f"{data_dir}/*{mode}*/*/{wav_id}.wav")
        if len(wav_file_) != 1:
            # print("Not found:", f"{DATA_DIR}/**/{wav_id}.wav")
            pass
        else:
            wav_file = wav_file_[0]
            wav_files.append(wav_file)

    print("Found files:", len(wav_files), '/', len(wav_ids))
    return wav_files

def audioset_valid_boundary(end_thre, start_time, end_time):
    if end_thre <= start_time: return False
    time_length = end_time - start_time
    if time_length < 1e-8: return False
    flag1 = ((end_thre - start_time) / time_length) > 0.5
    return flag1

def audioset_get_unique_classes(data, axis):
    result = [d[axis] for d in data]
    result = np.unique(np.array(result)).tolist()
    return result

def audioset_convert_to_seg(audio_file):
    global weak_seg_df
    global strong_seg_df

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

    (audio, _) = librosa.core.load(audio_file, sr=PARAMS['sample_rate'], mono=True)
    # audio = pad_or_truncate(audio, params['clip_samples'])

    pq = PriorityQueue()
    wav_meta = strong_temp_data.sort_values(by='start_time_seconds')
    max_time = wav_meta['end_time_seconds'].max()
    meta_index, new_wav_id = 0, 0

    start_time_list = wav_meta['start_time_seconds'].tolist()
    end_time_list = wav_meta['end_time_seconds'].tolist()
    class_label_list = wav_meta['label_id'].tolist()
    
    for start_time in np.arange(0, max_time, PARAMS['crop_second']):
        end_time = start_time + PARAMS['crop_second']
    
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
            if meta_index < len(start_time_list) and audioset_valid_boundary(end_time, 
                start_time_list[meta_index], end_time_list[meta_index]): 
                pq.push(class_label_list[meta_index], end_time_list[meta_index])
                meta_index += 1
            else:
                break

        class_labels = audioset_get_unique_classes(pq._queue, axis=-1)
        if len(class_labels) == 0: continue
        class_label = int(np.isin(class_labels, POSITIVE_LABELS).any())

        if (not strong_include_pos_flag) and np.random.randint(10) != 0: continue

        # if start_time - PARAMS['croped_pad_second'] < 0:
        #     start_time = start_time
        #     end_time = end_time + PARAMS['croped_pad_second'] * 2
        # else:
        #     start_time = start_time - PARAMS['croped_pad_second']
        #     end_time = end_time + PARAMS['croped_pad_second']

        start_sample_point = int(start_time * PARAMS['sample_rate'])
        end_sample_point = int(end_time * PARAMS['sample_rate'])
        sub_wav = audio[start_sample_point:end_sample_point]

        num_smaple_point_ = sub_wav.shape[0]
        if num_smaple_point_ < PARAMS['sample_rate'] // 2: continue

        # num_block_point_  = (PARAMS['crop_second'] + 0.1) * PARAMS['sample_rate']
        num_block_point_  = PARAMS['crop_second'] * PARAMS['sample_rate']
        if num_smaple_point_ < num_block_point_:
            sub_wav = np.pad(sub_wav, (0, int(num_block_point_ - num_smaple_point_)))

        save_wav_file = os.path.join(save_dir, f"{'bark' if class_label else 'background'}" \
                                                f".{wav_id}_{new_wav_id}.wav")
        soundfile.write(save_wav_file, sub_wav, PARAMS['sample_rate'])

        new_wav_id += 1
    

def process_audioset():
    global PROCESS_MODE
    global WAV_DIR_NAME
    global PARAMS
    global IGNORE_LABELS
    global POSITIVE_LABELS
    global strong_seg_df
    global weak_seg_df
        
    DATA_DIR = '/media/ubuntu/HD_new/download/audioset/audios'

    POSITIVE_CLSSSES = ['Bark', 'Bow-wow', 'Yip']
    IGNORE_CLASSES = ['Dog', 'Howl', 'Growling', 'Whimper (dog)']

    CLASS2LABEL, MID2LABEL = {}, {}

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
    
    IGNORE_LABELS = [CLASS2LABEL[x] for x in IGNORE_CLASSES]
    POSITIVE_LABELS = [CLASS2LABEL[x] for x in POSITIVE_CLSSSES]

    wav_files = audioset_get_all_strong_wavs(DATA_DIR, strong_seg_df, PROCESS_MODE)

    with Pool(16) as pool:
        pool.map(audioset_convert_to_seg, wav_files)


def urbansound8k_convert_to_seg(audio_file):
    global SAVE_DIR
    global WAV_DIR_NAME
    global PROCESS_MODE
    global strong_seg_df

    save_dir = f"{SAVE_DIR}/{WAV_DIR_NAME}/{PROCESS_MODE}"
    os.makedirs(save_dir, exist_ok=True)

    wav_id = os.path.splitext(os.path.basename(audio_file))[0]
    (audio, _) = librosa.core.load(audio_file, sr=PARAMS['sample_rate'], mono=True)
    # audio = pad_or_truncate(audio, params['clip_samples'])
    max_time = audio.shape[0] / PARAMS['sample_rate']
    meta_index, new_wav_id = 0, 0

    for start_time in np.arange(0, max_time, PARAMS['crop_second']):
        end_time = start_time + PARAMS['crop_second']

        # if start_time - PARAMS['croped_pad_second'] < 0:
        #     start_time = start_time
        #     end_time = end_time + PARAMS['croped_pad_second'] * 2
        # else:
        #     start_time = start_time - PARAMS['croped_pad_second']
        #     end_time = end_time + PARAMS['croped_pad_second']

        start_sample_point = int(start_time * PARAMS['sample_rate'])
        end_sample_point = int(end_time * PARAMS['sample_rate'])
        sub_wav = audio[start_sample_point:end_sample_point]

        num_smaple_point_ = sub_wav.shape[0]
        if num_smaple_point_ < PARAMS['crop_second'] * PARAMS['sample_rate'] // 4: continue

        # num_block_point_  = (PARAMS['crop_second'] + 0.1) * PARAMS['sample_rate']
        num_block_point_  = PARAMS['crop_second'] * PARAMS['sample_rate']
        if num_smaple_point_ < num_block_point_:
            sub_wav = np.pad(sub_wav, (0, int(num_block_point_ - num_smaple_point_)))

        if np.sum(np.fabs(sub_wav) > 0.1) / PARAMS['sample_rate'] < PARAMS['crop_second'] * 0.05:
            continue

        save_wav_file = os.path.join(save_dir, f"bark.{wav_id}_{new_wav_id}.wav")
        soundfile.write(save_wav_file, sub_wav, PARAMS['sample_rate'])

        new_wav_id += 1

def urbansound8k_get_all_strong_wavs(data_dir, strong_seg_df):
    global META_DIR
    
    non_ignore_wav_files = []
    ignore_wavs_xml = XMLParse(f"{META_DIR}/UrbanSound8K/ignore_wavs.xspf")
    ignore_wavs_xml_root = ignore_wavs_xml.documentElement   
    trackList = ignore_wavs_xml_root.getElementsByTagName('trackList')[0] 
    trackList = trackList.getElementsByTagName('track')
    for track_item in trackList:
        location_item = track_item.getElementsByTagName('location')[0]
        ignore_wav_file = location_item.firstChild.data.split('/')[-1]
        non_ignore_wav_files.append(ignore_wav_file)

    wav_files = []
    strong_temp_data = strong_seg_df[strong_seg_df['class'] == 'dog_bark']
    if strong_temp_data.shape[0] == 0:
        return wav_files
    
    for i, item_strong_data in strong_temp_data.iterrows():
        kfold = item_strong_data['fold']
        slice_file_name = item_strong_data['slice_file_name']
        if slice_file_name not in non_ignore_wav_files: continue
        wav_files.append(f"{data_dir}/fold{kfold}/{slice_file_name}")
    
    return wav_files
        
def process_urbansound8k():
    global strong_seg_df

    DATA_DIR = '/media/ubuntu/HD/Data/UrbanSound8K/UrbanSound8K/audio'

    strong_seg_df = pd.read_csv(f"{META_DIR}/UrbanSound8K/UrbanSound8K.csv")

    if PROCESS_MODE == 'train':
        train_kfold = list(range(1, 10))
        strong_seg_df = strong_seg_df[strong_seg_df['fold'].isin(train_kfold)]
    elif PROCESS_MODE == 'eval':
        eval_kfold = [10]
        strong_seg_df = strong_seg_df[strong_seg_df['fold'].isin(eval_kfold)]

    wav_files = urbansound8k_get_all_strong_wavs(DATA_DIR, strong_seg_df)
    
    with Pool(16) as pool:
        pool.map(urbansound8k_convert_to_seg, wav_files)


if __name__ == '__main__':
    # process_audioset()
    # process_urbansound8k()