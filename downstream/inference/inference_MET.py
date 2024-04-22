import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import dc, hd95
from multiprocessing.pool import Pool
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd


def compute_BraTS_dice(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    :param ref:
    :param gt:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 1
        else:
            return 0
    else:
        return dc(pred, ref)


def compute_BraTS_HD95(ref, pred):
    """
    ref and gt are binary integer numpy.ndarray s
    spacing is assumed to be (1, 1, 1)
    :param ref:
    :param pred:
    :return:
    """
    num_ref = np.sum(ref)
    num_pred = np.sum(pred)

    if num_ref == 0:
        if num_pred == 0:
            return 0
        else:
            return 373.12866
    elif num_pred == 0 and num_ref != 0:
        return 373.12866
    else:
        return hd95(pred, ref, (1, 1, 1))


def evaluate_BraTS_case(arr: np.ndarray, arr_gt: np.ndarray):
    """
    attempting to reimplement the brats evaluation scheme

    ET=3, ED=2, NCR=1
    ET = ET 
    TC = ET(3) + NCR(1)
    WT = ED(2) + TC(1+3)
    :param arr:
    :param arr_gt:
    :return:
    """
    # whole tumor
    mask_gt = (arr_gt != 0).astype(int)
    mask_pred = (arr != 0).astype(int)
    dc_whole = compute_BraTS_dice(mask_gt, mask_pred)
    hd95_whole = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # tumor core
    mask_gt = (arr_gt == 1).astype(int) + (arr_gt == 3).astype(int)
    mask_pred = (arr == 1).astype(int) + (arr == 3).astype(int)
    dc_core = compute_BraTS_dice(mask_gt, mask_pred)
    hd95_core = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    # enhancing
    mask_gt = (arr_gt == 3).astype(int)
    mask_pred = (arr == 3).astype(int)
    dc_enh = compute_BraTS_dice(mask_gt, mask_pred)
    hd95_enh = compute_BraTS_HD95(mask_gt, mask_pred)
    del mask_gt, mask_pred

    return dc_whole, dc_core, dc_enh, hd95_whole, hd95_core, hd95_enh


def load_evaluate(filename_gt: str, filename_pred: str):
    arr_pred = sitk.GetArrayFromImage(sitk.ReadImage(filename_pred))
    arr_gt = sitk.GetArrayFromImage(sitk.ReadImage(filename_gt))
    return evaluate_BraTS_case(arr_pred, arr_gt)


def evaluate_BraTS_folder(folder_pred, folder_gt, num_processes: int = 24, strict=False):
    nii_pred = subfiles(folder_pred, suffix='.nii.gz', join=False)
    if len(nii_pred) == 0:
        return
    nii_gt = subfiles(folder_gt, suffix='.nii.gz', join=False)
    for i in nii_pred:
        if i not in nii_gt:
            print (i)
    assert all([i in nii_gt for i in nii_pred]), 'not all predicted niftis have a reference file!'
    if strict:
        assert all([i in nii_pred for i in nii_gt]), 'not all gt niftis have a predicted file!'
    p = Pool(num_processes)
    nii_pred_fullpath = [join(folder_pred, i) for i in nii_pred]
    nii_gt_fullpath = [join(folder_gt, i) for i in nii_pred]
    results = p.starmap(load_evaluate, zip(nii_gt_fullpath, nii_pred_fullpath))
    # now write to output file
    with open(join(folder_pred, 'results.csv'), 'w') as f:
        f.write("name,dc_whole,dc_core,dc_enh,hd95_whole,hd95_core,hd95_enh\n")
        for fname, r in zip(nii_pred, results):
            f.write(fname)
            f.write(",%0.4f,%0.4f,%0.4f,%3.3f,%3.3f,%3.3f\n" % r)


def main():
    # folder_pred = '/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/GIL_nn'
    # evaluate_BraTS_folder(folder_pred,
    #                       '/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task999_BraTS2021/labelsTs')

    # folder_pred = '/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/MET_ft_BN'
    # evaluate_BraTS_folder(folder_pred,
    #                       '/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/labelsTs')

    # folder_pred = '/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/SSA_nn'
    # evaluate_BraTS_folder(folder_pred,
    #                       '/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task995_BraTS2023_SSA/labelsTs')
    
    # folder_pred = '/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/PED_nn'
    # evaluate_BraTS_folder(folder_pred,
    #                       '/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task996_BraTS2023_PED/labelsTs')
    
    folder_pred = '/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/nomod_23'
    evaluate_BraTS_folder(folder_pred,
                          '/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/labelsTs')

    result = pd.read_csv(join(folder_pred, 'results.csv'))[
        ['dc_whole', 'dc_core', 'dc_enh', 'hd95_whole', 'hd95_core', 'hd95_enh']]
    print(result.mean(axis=0))


if __name__ == '__main__':
    main()
