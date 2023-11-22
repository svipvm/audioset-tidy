import pandas as pd
import numpy as np
import glob, os, librosa, soundfile, json, h5py, logging

import librosa.display
from multiprocessing import Pool

from utils.util_process import *

DATA_DIR = '/media/ubuntu/ssd2t/AIGroup/Audio-Data/audioset_strong'
# LOGMEL_DIR_NAMES = ['audioset_logmel_16k', 'urbansound8k_logmel_16k', 'plain_logmel_16k']
# LOGMEL_DIR_NAMES = ['audioset_logmel_32k', 'urbansound8k_logmel_32k', 'plain_logmel_32k']
LOGMEL_DIR_NAMES = ['audioset_logmel_16k', 'urbansound8k_logmel_16k']
# LOGMEL_DIR_NAMES = ['audioset_logmel_32k', 'urbansound8k_logmel_32k']
PROCESS_MODE = 'eval'

assert PROCESS_MODE in ('train', 'eval')

def combine_indexes(hdf5_files, index_file):
    os.makedirs(os.path.dirname(index_file), exist_ok=True)
    with h5py.File(index_file, 'w') as full_hf:
        full_hf.create_dataset(name='audio_name', shape=(0,), maxshape=(None,), dtype='S20')
        full_hf.create_dataset(name='target', shape=(0,), maxshape=(None,), 
                               dtype=h5py.special_dtype(vlen=np.dtype('int32')))
        full_hf.create_dataset(name='hdf5_path', shape=(0,), maxshape=(None,), dtype='S200')
        full_hf.create_dataset(name='index_in_hdf5', shape=(0,), maxshape=(None,), dtype=np.int32)

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
    
    print('Write combined full hdf5 to {}'.format(index_file))
    
if __name__ == '__main__':
    index_file = f"{DATA_DIR}/hdf5s/{PROCESS_MODE}_index.h5"
    hdf5_files = [f"{DATA_DIR}/{logmel_dir}/{PROCESS_MODE}_index.h5" for logmel_dir in LOGMEL_DIR_NAMES]
    combine_indexes(hdf5_files, index_file)