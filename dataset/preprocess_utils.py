import shutil
import subprocess
import matplotlib.pyplot as plt
from pathlib import Path
# from multiprocessing import Pool, cpu_count
import os
import logging
from nipype.interfaces.ants import N4BiasFieldCorrection
from nilearn.plotting import plot_anat
from scipy.signal import medfilt
from sklearn.cluster import KMeans
import numpy as np
import nibabel as nib

def showImg(img, title):
    plot_anat(img, title=title, display_mode='ortho', dim=-1, draw_cross=False, annotate=False)
    plt.show()

def create_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)

def plotMiddle(data, slice_no=None):
    if not slice_no:
        slice_no = data.shape[-1] // 2
    plt.figure()
    plt.imshow(data[..., slice_no], cmap="gray")
    plt.show()
    return

def runACPCDetect(niifile, acpcDetectPath='./utils/acpcdetect/bin/acpcdetect'):
    command = [acpcDetectPath, "-no-tilt-correction", "-center-AC", "-nopng", "-noppm", "-i", niifile]
    subprocess.call(command, stdout=open(os.devnull, "r"), stderr=subprocess.STDOUT)


def orient2std(src_path, dst_path):
    command = ["fslreorient2std", src_path, dst_path]
    subprocess.call(command)
    return

def registration(src_path, dst_path, ref_path):
    command = ["flirt", "-in", src_path, "-ref", ref_path, "-out", dst_path,
               "-bins", "256", "-cost", "corratio", "-searchrx", "0", "0",
               "-searchry", "0", "0", "-searchrz", "0", "0", "-dof", "12",
               "-interp", "spline"]
    subprocess.call(command, stdout=open(os.devnull, "r"), stderr=subprocess.STDOUT)
    return

def unwarp_main_registration(arg, **kwarg):
    return main_registration(*arg, **kwarg)

def main_registration(src_path, dst_path, ref_path):
    print("Registration on: ", src_path)
    try:
        orient2std(src_path, dst_path)
        registration(dst_path, dst_path, ref_path)
    except RuntimeError:
        print("\tFalied on: ", src_path)

    return


def load_nii(path):
    nii = nib.load(path)
    return nii.get_data(), nii.affine


def save_nii(data, path, affine):
    nib.save(nib.Nifti1Image(data, affine), path)
    return


def denoise(volume, kernel_size=3):
    return medfilt(volume, kernel_size)


def rescale_intensity(volume, percentils=[0.5, 99.5], bins_num=0):
    obj_volume = volume[np.where(volume > 0)]
    min_value = np.percentile(obj_volume, percentils[0])
    max_value = np.percentile(obj_volume, percentils[1])

    if bins_num == 0:
        obj_volume = (obj_volume - min_value) / (max_value - min_value).astype(np.float32)
    else:
        obj_volume = np.round((obj_volume - min_value) / (max_value - min_value) * (bins_num - 1))
        obj_volume[np.where(obj_volume < 1)] = 1
        obj_volume[np.where(obj_volume > (bins_num - 1))] = bins_num - 1

    volume = volume.astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume

    return volume


def equalize_hist(volume, bins_num=256):
    obj_volume = volume[np.where(volume > 0)]
    # hist, bins = np.histogram(obj_volume, bins_num, normed=True)
    hist, bins = np.histogram(obj_volume, bins_num)
    cdf = hist.cumsum()
    cdf = (bins_num - 1) * cdf / cdf[-1]

    obj_volume = np.round(np.interp(obj_volume, bins[:-1], cdf)).astype(obj_volume.dtype)
    volume[np.where(volume > 0)] = obj_volume
    return volume


def enhance(src_path, dst_path, kernel_size=3,
            percentils=[0.5, 99.5], bins_num=256, eh=True):
    logging.info('Preprocess on: {}'.format(src_path))
    try:
        volume, affine = load_nii(src_path)
        volume = denoise(volume, kernel_size)
        volume = rescale_intensity(volume, percentils, bins_num)
        if eh:
            volume = equalize_hist(volume, bins_num)
        save_nii(volume, dst_path, affine)
    except RuntimeError:
        logging.warning('Failed on: {}'.format(src_path))



if __name__ == '__main__':

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')

    # Reference FSL sample template
    # export FSLDIR=/usr/local/fsl
    refPath = '$FSLDIR/data/standard/MNI152_T1_1mm.nii.gz'

    # Set ART location
    os.environ['ARTHOME'] = '/usr/local/art'

    # Set ACPC Path
    acpcDetectPath = './utils/acpcdetect/bin/acpcdetect'

    # ACPC detection

    niiPaths = Path('../dataset_/data_train_plus_test_sourceres/train/source_resolution/Train').glob('*.nii.gz')

    niiFiles = [niiPath for niiPath in niiPaths if niiPath.is_file()]

    for niiFile in niiFiles:
        runACPCDetect(niiFile, acpcDetectPath)

    # Zip _RAS.nii files.
    niiACPCPaths = Path('./data').glob('**/*_RAS.nii')
    niiACPCFiles = [niiACPCPath for niiACPCPath in niiACPCPaths if niiACPCPath.is_file()]

    for niiFile in niiACPCFiles:
        dstFilePath = niiFile.parent / ('_' + niiFile.name)
        shutil.copyfile(niiFile, dstFilePath)
        subprocess.check_call(['gzip', dstFilePath, '-f'])

    # Run orient2std and registration

    # niiGzPaths = Path('./dataset_/data_train_plus_test_source_resolution/train/source_resolution/Train').glob('**/*.gz')
    # niiGzFiles = [niiGzPath for niiGzPath in niiGzPaths if niiGzPath.is_file()]

    # regFiles = list()
    # for niiGzFile in niiGzFiles:
    #     niiGzFilePath = niiGzFile.as_posix()
    #     dstFile = niiGzFile.parent / (niiGzFile.stem.split('.')[0] + '_reg.nii.gz')
    #     regFiles.append(dstFile)
    #     dstFilePath = dstFile.as_posix()
    #     logging.info('Registration on: {}'.format(niiGzFilePath))
    #     try:
    #         orient2std(niiGzFilePath, dstFilePath)
    #         registration(dstFilePath, dstFilePath, refPath)
    #     except RuntimeError:
    #         logging.warning('Falied on: {}'.format(niiGzFilePath))

    # Skull stripping

    # stripFiles = list()
    # for regFile in regFiles:
    #     regFilePath = regFile.as_posix()
    #     dstFile = regFile.parent / (regFile.stem.split('.')[0] + '_strip.nii.gz')
    #     stripFiles.append(dstFile)
    #     dstFilePath = dstFile.as_posix()
    #     logging.info('Stripping on : {}'.format(regFilePath))
    #     try:
    #         bet(regFilePath, dstFilePath, frac=0.3)
    #     except RuntimeError:
    #         logging.warning('Failed on: {}'.format(regFilePath))
    #
    # # Bias correction
    # bcFiles = list()
    # for stripFile in stripFiles:
    #     stripFilePath = stripFile.as_posix()
    #     dstFile = stripFile.parent / (stripFile.stem.split('.')[0] + '_bc.nii.gz')
    #     bcFiles.append(dstFile)
    #     dstFilePath = dstFile.as_posix()
    #     bias_field_correction(stripFilePath, dstFilePath)

    # # Enhancement
    # enhancedFiles = list()
    # for bcFile in bcFiles:
    #     print('hi')
    #     bcFilePath = bcFile.as_posix()
    #     dstFile = bcFile.parent / (bcFile.stem.split('.')[0] + '_eh.nii.gz')
    #     enhancedFiles.append(dstFile)
    #     dstFilePath = dstFile.as_posix()
    #     enhance(bcFilePath, dstFilePath, kernel_size=3, percentils=[0.5, 99.5], bins_num=256, eh=True)

    # Segmentation
    # segmentFiles = list()
    #
    # for enhancedFile in enhancedFiles:
    #     enhancedFilePath = enhancedFile.as_posix()
    #     dstFile = enhancedFile.parent / (enhancedFile.stem.split('.')[0].split('_')[0] + '_segment.nii.gz')
    #     labelFile = enhancedFile.parent / (enhancedFile.stem.split('.')[0].split('_')[0] + '_segment_labeled.nii.gz')
    #     segmentFiles.append(dstFile)
    #     dstFilePath = dstFile.as_posix()
    #     labelFilePath = labelFile.as_posix()
    #     segment(enhancedFilePath, dstFilePath, labels_path=labelFilePath)

