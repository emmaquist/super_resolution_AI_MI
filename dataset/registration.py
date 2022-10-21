from __future__ import print_function
from preprocess_utils import *
from multiprocessing import Pool, cpu_count

#  export ANTSPATH=/home/emmq/install/bin/
# export PATH=${ANTSPATH}:$PATH

# Reference FSL sample template
# export FSLDIR=/usr/local/fsl
refPath = '$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz'

# reg_path = "../dataset_/Template/MNI152_T1_1mm_brain.nii.gz"
# ref_path = "../dataset_/Template/MNI152_T1_1mm.nii.gz"

parent_dir = os.path.dirname(os.getcwd())

data_dir = os.path.join(parent_dir, "dataset/data_train_plus_test_source_resolution")
data_dst_dir = os.path.join(data_dir + "_reg")
create_dir(data_dst_dir)
# create_dir(data_dst_dir_reg)

data_src_paths, data_dst_paths = [], []
for t in ['test', 'train']:
    data_t_dir = os.path.join(data_dir, t)
    data_t_dst_dir = os.path.join(data_dst_dir, t + "_reg")
    create_dir(data_t_dst_dir)
    for res in ['source_resolution', 'target_resolution']:
        if t == "train":
            data_res_dir = os.path.join(data_t_dir, res)
            data_res_dst_dir = os.path.join(data_t_dst_dir, res + "_reg")
            create_dir(data_res_dst_dir)
            for t_v in ["Train", "Val"]:
                data_res_dir_tv = os.path.join(data_res_dir, t_v)
                data_res_dst_dir_tv = os.path.join(data_res_dst_dir, t_v + "_reg")
                create_dir(data_res_dst_dir_tv)
                for subject in os.listdir(data_res_dir_tv):
                    data_src_paths.append(os.path.join(data_res_dir_tv, subject))
                    data_dst_paths.append(os.path.join(data_res_dst_dir_tv, subject))
        else:
            data_res_dir = os.path.join(data_t_dir, res)
            data_res_dst_dir = os.path.join(data_t_dst_dir, res + "_reg")
            create_dir(data_res_dst_dir)
            for subject in os.listdir(data_res_dir):
                data_src_paths.append(os.path.join(data_res_dir, subject))
                data_dst_paths.append(os.path.join(data_res_dst_dir, subject))


"""
Registration
"""
#test
# main_registration(data_src_paths[0], data_dst_paths[0], refPath)

#Multi-processing
paras = zip(data_src_paths, data_dst_paths,
            [refPath] * len(data_src_paths))
pool = Pool(processes=cpu_count())
pool.map(unwarp_main_registration, paras)

# T1_orig = "../dataset_/data_train_plus_test_source_resolution/train/source_resolution/Train/SET_A_10.nii.gz"
# T1_corrected = "../dataset_/data_train_plus_test_source_resolution/train/source_resolution/TrainDenoise/SET_A_10.nii.gz"
# T1_reg = "../dataset_/data_train_plus_test_source_resolution/train/source_resolution/TrainDenoise/SET_A_10.nii.gz"
# T1_img_orig = nib.load(T1_orig)
# T1_img_corrected = nib.load(T1_corrected)
# T1_img_reg = nib.load(T1_reg)
#
# plotMiddle(T1_img_orig.get_data(), slice_no=None)
#
# plotMiddle(T1_img_corrected.get_data(), slice_no=None)
#
# plotMiddle(T1_img_reg.get_data(), slice_no=None)

# print(T1_img_orig.get_data()[45][T1_img_orig.get_data()[45] == T1_img_corrected.get_data()[45]])