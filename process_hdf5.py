import h5py, glob, tifffile, os, json, random
from tqdm import tqdm
import numpy as np
import pandas as pd

DATA_DIR = '/media/ubuntu/HD/Data/Audioset-Seg/data_logmel'
SAVE_DIR = '/media/ubuntu/HD/Data/Audioset-Seg/data_logmel_hdf5'

BATCH_SIZE = 1024
INPUT_SHAPE = (65, 64)

if __name__ == '__main__':
    metadata = pd.read_csv('./metadata/tiff_metadata.csv')
    classes = metadata['class'].unique()
    classes = np.sort(classes)
    class2label = {class_name: class_label for class_label, class_name in enumerate(classes)}
    with open('./metadata/tiff_class2label.json', 'w') as f:
        f.write(json.dumps(class2label))
    fold_ids = metadata['fold'].unique()

    random.seed(42)

    for fold_id in tqdm(fold_ids):
        temp_data = metadata[metadata['fold'] == fold_id].copy()
        # temp_data = temp_data.sample(frac=1, random_state=42)

        tiff_files = []
        for _, item_data in temp_data.iterrows():
            class_name = item_data['class']
            tiff_id = item_data['id'] 
            tiff_files += glob.glob(f'{DATA_DIR}/{class_name}/{tiff_id}_*')
            # if len(tiff_files) > 10000: break

        print(fold_ids, len(tiff_files))
        random.shuffle(tiff_files)
        
        for i in range(0, len(tiff_files), BATCH_SIZE):
            start_i = i
            end_i = min(i + BATCH_SIZE, len(tiff_files))

            output_file = f'{SAVE_DIR}/fold{fold_id}/data_{start_i}_{end_i}.h5'
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # print(output_file)
            with h5py.File(output_file, "w") as h5_file:
                image_dataset = h5_file.create_dataset("datasets", (end_i - start_i, *INPUT_SHAPE), 
                                           dtype=np.float16)
                label_dataset = h5_file.create_dataset("labels", (end_i - start_i, ), 
                                                    dtype=np.int32)
                for h_i, image_file in enumerate(tiff_files[start_i: end_i]):
                    image_data = tifffile.imread(image_file)
                    class_name = image_file.split('/')[-2]
                    image_dataset[h_i] = image_data.astype(np.float16)
                    label_dataset[h_i] = class2label[class_name]
        # break
    print('Finished!')
