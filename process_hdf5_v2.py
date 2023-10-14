import h5py, glob, tifffile, os, json, random
from tqdm import tqdm
import numpy as np
import pandas as pd

DATA_DIR = '/media/ubuntu/HD/Data/Audioset-Seg/data_cut_logmel_2.0s'
SAVE_DIR = f'{DATA_DIR}_hdf5'
META_FILE = '/media/ubuntu/HD/Data/Audioset-Seg/metadata/cross_valid_2s.csv'

num_classes = 456
INPUT_SHAPE = (201, 64)

# def convert2onehot(labels):
#     one_hot_encoding = np.zeros(num_classes, dtype=np.int32)
#     one_hot_encoding[labels] = 1
#     # print(labels, one_hot_encoding)
#     return one_hot_encoding

if __name__ == '__main__':
    metadata = pd.read_csv(META_FILE)
    fold_ids = metadata['kfold'].unique()
    metadata['h5_index'] = [-1] * metadata.shape[0]

    random.seed(42)

    for fold_id in tqdm(fold_ids):
        temp_data = metadata[metadata['kfold'] == fold_id].copy()
        # temp_data = temp_data.sample(frac=1, random_state=42)

        tiff_files, tiff_lables = [], []
        for _, item_data in temp_data.iterrows():
            seg_part = item_data['segments']
            tiff_id = item_data['wav_id']
            wav_id = '_'.join(tiff_id.split('_')[:-1])
            class_labels = item_data['classes']
            tiff_files.append(f'{DATA_DIR}/{seg_part}/{wav_id}/{tiff_id}.tiff')
            tiff_lables.append(eval(class_labels))
            # if len(tiff_files) > 1000: break # debug

        print('kfold:', fold_id, ' wav file:', len(tiff_files))
        shuffle_index = np.random.permutation(len(tiff_files))
        metadata.loc[temp_data.index[shuffle_index], 'h5_index'] = list(range(temp_data.shape[0]))
        tiff_files = [tiff_files[i] for i in shuffle_index]
        tiff_lables = [tiff_lables[i] for i in shuffle_index]
        num_tiff = len(tiff_files)

        output_file = f'{SAVE_DIR}/fold{fold_id}_data_pack.h5'
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # print(output_file)
        with h5py.File(output_file, "w") as h5_file:
            image_dataset = h5_file.create_dataset("datasets", (num_tiff, *INPUT_SHAPE), 
                                        dtype=np.float16)
            label_dataset = h5_file.create_dataset("labels", (num_tiff, ), 
                                        dtype=h5py.special_dtype(vlen=np.dtype('int32')))

            data_i = 0
            for h_i, (image_file, image_labels) in enumerate(
                    zip(tiff_files, tiff_lables)):
                # if len(image_labels) <= 1: continue
                image_data = tifffile.imread(image_file)
                class_name = image_file.split('/')[-2]
                image_dataset[data_i] = image_data.astype(np.float16)
                label_dataset[data_i] = image_labels
                data_i += 1
                # label_dataset[h_i] = convert2onehot(image_labels)
        # break # debug
    metadata.to_csv(f"{os.path.dirname(META_FILE)}/train_hdf5_2s.csv", index=False)
    print('Finished!')
