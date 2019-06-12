import numpy as np
import nibabel as nib
import os

def read_scan(img_file_path):
    if 'defaced' in img_file_path:
        label = 1
    else:
        label = 0
    img = nib.load(img_file_path)
    return img.get_data(), img.affine, label

def process_mean(img):
    mean_0 = np.mean(img,axis=0)
    mean_1 = np.mean(img,axis=1)
    mean_2 = np.mean(img,axis=2)
    return mean_0, mean_1, mean_2

def process_slice(img):
    all_dimensions = img.shape
    slice_0 = img[int(all_dimensions[0]/2),:,:]
    slice_1 = img[:,int(all_dimensions[1]/2),:]
    slice_2 = img[:,:,int(all_dimensions[2]/2)]
    return slice_0, slice_1, slice_2

def save_preprocessed_files(folder_path, output_folder_path, preprocessing):
    valid_flag = 1
    i = 1
    all_files_list = sorted(os.listdir(folder_path))
    for each_file in all_files_list:
        if each_file.endswith('.nii.gz'):
            each_file_path = os.path.join(folder_path, each_file)
            data, affine, label = read_scan(each_file_path)
            if label == 1:
                file_prefix = str(i) + '_defaced_' + preprocessing
                sub_dir = 'defaced'
            else:
                file_prefix = str(i) + '_original_' + preprocessing
                sub_dir = 'original'
            if preprocessing == 'mean':
                d1, d2, d3 = process_mean(data)
            elif preprocessing == 'slice':
                d1, d2, d3 = process_slice(data)
            else:
                valid_flag = 0
                break
            each_img_out_path = os.path.join(output_folder_path,str(i),sub_dir,preprocessing)
            os.makedirs(each_img_out_path, exist_ok=True)
            nib.save(nib.Nifti1Image(d1,affine),os.path.join(each_img_out_path, file_prefix+'_0.nii.gz'))
            nib.save(nib.Nifti1Image(d2,affine),os.path.join(each_img_out_path, file_prefix+'_1.nii.gz'))
            nib.save(nib.Nifti1Image(d3,affine),os.path.join(each_img_out_path, file_prefix+'_2.nii.gz'))
            i+=1

    if not valid_flag:
        print('preprocessing must be [mean] OR [slice]')

def prepare_preprocessed_dataset(output_folder_path):
    input_folder_path = ['IXI-T1', 'IXI-T1-Defaced']
    preprocessing_options = ['mean', 'slice']
    for each_input_folder in input_folder_path:
        for each_preprocessing_option in preprocessing_options:
            save_preprocessed_files(each_input_folder, output_folder_path, each_preprocessing_option)

prepare_preprocessed_dataset('IXI-T1-Preprocessed')
