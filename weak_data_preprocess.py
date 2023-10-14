import pandas as pd
import numpy as np
import glob, os, librosa, soundfile, json, h5py, logging

import librosa.display
from multiprocessing import Pool

from utils.util_process import *

DATA_DIR = '/media/ubuntu/HD_new/download/audioset/audios'
META_DIR = '/media/ubuntu/HD_new/download/audioset/metadata'
SAVE_DIR = '/media/ubuntu/ssd2t/AIGroup/Audio-Data/audioset'

MINI_MODE = False
PROCESS_MODE = 'unbalanced_train'
AUDIO_DIR = glob.glob(f"{DATA_DIR}/{PROCESS_MODE}_segments/unbalanced*")

params = {
    'input_size': (1001, 64),
    'sample_rate': 32000,
    'clip_samples': 32000 * 10, 
    'n_fft': 1024,
    'hop_length': 320,
    'win_length': 1024,
    'lower_hertz': 50,
    'upper_hertz': 14000,
    'mel_bins': 64
}

def read_metadata(csv_path, id_to_ix):
    """Read metadata of AudioSet from a csv file.

    Args:
      csv_path: str

    Returns:
      meta_dict: {'audio_name': (audios_num,), 'target': (audios_num, classes_num)}
    """

    with open(csv_path, 'r') as fr:
        lines = fr.readlines()
        lines = lines[3:]   # Remove heads

    targets = []
    audio_names = []
 
    for n, line in enumerate(lines):
        items = line.split(', ')
        """items: ['--4gqARaEJE', '0.000', '10.000', '"/m/068hy,/m/07q6cd_,/m/0bt9lr,/m/0jbk"\n']"""

        audio_name = 'Y{}.wav'.format(items[0])   # Audios are started with an extra 'Y' when downloading
        label_ids = items[3].split('"')[1].split(',')

        audio_names.append(audio_name)

        # Target
        targets.append([id_to_ix[id_] for id_ in label_ids])
        
    meta_dict = {'audio_name': audio_names, 'target': targets}
    return meta_dict

def pad_or_truncate(x, audio_length):
    """Pad all audio to specific length."""
    if len(x) <= audio_length:
        return np.concatenate((x, np.zeros(audio_length - len(x))), axis=0)
    else:
        return x[0 : audio_length]
    
def convert_to_hdf5(audio_dir):
    hdf5_file = f"{SAVE_DIR}/{PROCESS_MODE}/{os.path.basename(audio_dir)}.h5"
    wav_files = glob.glob(f"{audio_dir}/*")
    if MINI_MODE:
        wav_files = wav_files[:1234]
        hdf5_file = f"{SAVE_DIR}/{PROCESS_MODE}_mini/{os.path.basename(audio_dir)}.h5"
    os.makedirs(os.path.dirname(hdf5_file), exist_ok=True)
    audios_num = len(wav_files)
    
    with h5py.File(hdf5_file, 'w') as hf:
        hf.create_dataset('audio_name', shape=((audios_num,)), dtype='S20')
        hf.create_dataset('logmel', shape=((audios_num, *params['input_size'])), dtype=np.float32)
        hf.create_dataset('target', shape=((audios_num, )), dtype=
                                h5py.special_dtype(vlen=np.dtype('int32')))
        hf.attrs.create('sample_rate', data=params['sample_rate'], dtype=np.int32)

        # Pack waveform & target of several audio clips to a single hdf5 file
        for n, wav_file in enumerate(wav_files):
            audio_path = os.path.join(audio_dir, wav_file)
            # break
            if not os.path.isfile(audio_path):
                logging.info('{} File does not exist! {}'.format(n, audio_path))
                continue

            try:
                (audio, _) = librosa.core.load(audio_path, sr=params['sample_rate'], mono=True)
            except:
                hf['audio_name'][n] = '__delete__'
                continue
            audio = pad_or_truncate(audio, params['clip_samples'])
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
            audio_name = os.path.basename(wav_file)
            hf['audio_name'][n] = audio_name
            hf['logmel'][n] = logmel
            hf['target'][n] = meta_id_label_mapper[audio_name]
            
    print(audio_dir, 'has finished!')

def process_seg_wav():
    with Pool(16) as pool:
        pool.map(convert_to_hdf5, AUDIO_DIR)
    
    print("Finished!")

if __name__ == '__main__':
    print("start to convert hdf5...")
    class_labels_df = pd.read_csv(f'{META_DIR}/class_labels_indices.csv', delimiter=',')
    num_classes = class_labels_df.shape[0]

    labels = class_labels_df['display_name'].tolist()
    mid_list = class_labels_df['mid'].tolist()

    lb_to_ix = {label : i for i, label in enumerate(labels)}
    ix_to_lb = {i : label for i, label in enumerate(labels)}

    id_to_ix = {id : i for i, id in enumerate(mid_list)}
    ix_to_id = {i : id for i, id in enumerate(mid_list)}

    all_meta_dict = read_metadata(f"{META_DIR}/{PROCESS_MODE}_segments.csv", id_to_ix)
    audios_num = len(all_meta_dict['audio_name'])
    meta_id_label_mapper = {
        all_meta_dict['audio_name'][i]: all_meta_dict['target'][i] for i in range(audios_num)
    }
    with open("metadata/wav_id_labels.json", 'w') as f:
        f.write(json.dumps(meta_id_label_mapper))

    process_seg_wav()
