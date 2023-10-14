import pandas as pd
import numpy as np
import glob, os, librosa, soundfile, json, h5py, logging

import librosa.display
from multiprocessing import Pool

from utils.util_process import *

DATA_DIR = '/media/ubuntu/ssd2t/AIGroup/Audio-Data/audioset/unbalanced_train'
# DATA_DIR = '/media/ubuntu/ssd2t/AIGroup/Audio-Data/audioset/unbalanced_train_mini'
SAVE_DIR = '/media/ubuntu/ssd2t/AIGroup/Audio-Data/audioset/indexes'
np.random.seed(42)

def process_hdf5(data_file, index_file):
    num_kfold = 10
    with h5py.File(data_file, 'r') as hr:
        with h5py.File(index_file, 'w') as hw:
            valid_index = (hr['audio_name'][:] != b'__delete__')
            audios_num = sum(valid_index)

            # valid_index = (hr['audio_name'][:] == b'YDsiLRI_juuA.wav')
            # print(hr['audio_name'][:][valid_index])

            hw.create_dataset('audio_name', data=hr['audio_name'][:][valid_index], dtype='S20')
            hw.create_dataset('target', data=hr['target'][:][valid_index], 
                              dtype=h5py.special_dtype(vlen=np.dtype('int32')))
            hw.create_dataset('hdf5_path', data=[(data_file.replace(f"{DATA_DIR}/", '')).encode()
                                                 ] * audios_num, dtype='S200')
            hw.create_dataset('index_in_hdf5', data=np.arange(audios_num), dtype=np.int32)
            kfold_list = np.arange(0, audios_num, dtype=np.int32) % num_kfold
            np.random.shuffle(kfold_list)
            hw.create_dataset('kfold', data=kfold_list, dtype=np.int32)


def combine_indexes(hdf5_files, index_file):
    with h5py.File(index_file, 'w') as full_hf:
        full_hf.create_dataset(name='audio_name', shape=(0,), maxshape=(None,), dtype='S20')
        full_hf.create_dataset(name='target', shape=(0,), maxshape=(None,), 
                               dtype=h5py.special_dtype(vlen=np.dtype('int32')))
        full_hf.create_dataset(name='hdf5_path', shape=(0,), maxshape=(None,), dtype='S200')
        full_hf.create_dataset(name='index_in_hdf5', shape=(0,), maxshape=(None,), dtype=np.int32)
        full_hf.create_dataset(name='kfold', shape=(0,), maxshape=(None,), dtype=np.int32)

        for hdf5_file in tqdm(hdf5_files, desc='Combine'):
            with h5py.File(hdf5_file, 'r') as part_hf:
                n = len(full_hf['audio_name'][:])
                new_n = n + len(part_hf['audio_name'][:])

                full_hf['audio_name'].resize((new_n,))
                full_hf['audio_name'][n : new_n] = part_hf['audio_name'][:]

                full_hf['target'].resize((new_n,))
                full_hf['target'][n : new_n] = part_hf['target'][:]

                full_hf['hdf5_path'].resize((new_n,))
                full_hf['hdf5_path'][n : new_n] = part_hf['hdf5_path'][:]

                full_hf['index_in_hdf5'].resize((new_n,))
                full_hf['index_in_hdf5'][n : new_n] = part_hf['index_in_hdf5'][:]

                full_hf['kfold'].resize((new_n,))
                full_hf['kfold'][n : new_n] = part_hf['kfold'][:]
    
    print('Write combined full hdf5 to {}'.format(index_file))
    
    
if __name__ == '__main__':
    hdf5_files = glob.glob(f"{DATA_DIR}/*.h5")

    if os.path.isdir(f"{SAVE_DIR}/temp"):
        os.system(f"rm -rf {SAVE_DIR}/temp")

    temp_hdf5_files = []
    for hdf5_file in tqdm(hdf5_files, desc="Temp processer"):
        indexes_hdf5_path = f"{SAVE_DIR}/temp/{os.path.basename(hdf5_file)}"
        os.makedirs(os.path.dirname(indexes_hdf5_path), exist_ok=True)
        process_hdf5(hdf5_file, indexes_hdf5_path)
        temp_hdf5_files.append(indexes_hdf5_path)
        # break
    
    index_file = f"{SAVE_DIR}/{os.path.basename(DATA_DIR)}.h5"
    combine_indexes(temp_hdf5_files, index_file)

    if os.path.isdir(f"{SAVE_DIR}/temp"):
        os.system(f"rm -rf {SAVE_DIR}/temp")