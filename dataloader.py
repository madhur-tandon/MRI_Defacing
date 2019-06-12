import os
import nibabel as nib
import augmentation
import numpy as np

np.random.seed(42)

def get_data(folder_path, processing, dimension, target_size, train=True):
    data = []
    labels = []
    num_samples = 0
    all_files = os.listdir(folder_path)
    for each_file in all_files:
        if os.path.isdir(os.path.join(folder_path,each_file)):
            num_samples+=1
    if train:
        low = 0
        high = int(0.8*num_samples)
    else:
        low = int(0.8*num_samples)
        high = num_samples
    for i in range(low+1,high+1):
        file_name_original = str(i) + '_original_' + processing + '_{0}.nii.gz'.format(dimension)
        file_name_defaced = str(i) + '_defaced_' + processing + '_{0}.nii.gz'.format(dimension)
        file_path_original = os.path.join(folder_path,str(i),'original',processing,file_name_original)
        file_path_defaced = os.path.join(folder_path,str(i),'defaced',processing,file_name_defaced)
        data_original = nib.load(file_path_original).get_data()
        data_defaced = nib.load(file_path_defaced).get_data()
        data.append(data_original)
        labels.append(0)
        data.append(data_defaced)
        labels.append(1)

    data = augmentation.pipeline(data,(target_size,target_size))
    data = np.array(data)
    labels = np.array(labels)
    data = data.reshape(data.shape[0], -1)

    shuffled_indices = np.random.permutation(data.shape[0])
    shuffled_data = data[shuffled_indices]
    shuffled_labels = labels[shuffled_indices]

    return shuffled_data, shuffled_labels


            
