# MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='5' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' -outpath='UniMiss' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=5
# MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_SSA' -task='995' -outpath='UniMiss_multi_modal' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=5


# train from scratch UNETR
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MET' -task='997' -outpath='UNETR_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=100
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='2' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MET' -task='997' -outpath='UNETR_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=20
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='3' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MET' -task='997' -outpath='UNETR_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=10
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='4' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MET' -task='997' -outpath='UNETR_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=5

# train from scratch UniMiss
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='unimiss_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=100
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='1' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='unimiss_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=50
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='2' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='unimiss_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=10
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='3' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='unimiss_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=25


MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='4' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='unimiss_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/UniMiSS/checkpoint.pth' -split=100

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='5' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='unimiss_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/UniMiSS/checkpoint.pth' -split=50

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='6' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997'\
    -outpath='unimiss_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/UniMiSS/checkpoint.pth' -split=25

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='7' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='unimiss_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/UniMiSS/checkpoint.pth' -split=10

### MET

# fine-tune
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='5' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MET' -task='997' \
    -outpath='UNETR_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/MAE-pytorch/output/pretrain_mae_base_patch16_128/checkpoint-1999.pth' -split=100 \

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MET' -task='997' \
    -outpath='UNETR_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/MAE-pytorch/output/pretrain_mae_base_patch16_128/checkpoint-1999.pth' -split=20 \

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='1' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MET' -task='997'\
    -outpath='UNETR_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/MAE-pytorch/output/pretrain_mae_base_patch16_128/checkpoint-1999.pth' -split=10 \

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='2' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MET' -task='997' \
    -outpath='UNETR_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/MAE-pytorch/output/pretrain_mae_base_patch16_128/checkpoint-1999.pth' -split=5 \

# inference
CUDA_VISIBLE_DEVICES=1 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_0" -t 997 -tr TrainerV2_Brats23_UNETR_MET -chk model_best --overwrite_existing -split=100

CUDA_VISIBLE_DEVICES=2 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_1" -t 997 -tr TrainerV2_Brats23_UNETR_METT -chk model_best --overwrite_existing -split=20

CUDA_VISIBLE_DEVICES=3 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_2" -t 997 -tr TrainerV2_Brats23_UNETR_MET -chk model_best --overwrite_existing -split=10

CUDA_VISIBLE_DEVICES=4 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_3" -t 997 -tr TrainerV2_Brats23_UNETR_MET -chk model_best --overwrite_existing -split=5



### MEN

# train from scratch
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='5' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MEN' -task='998' \
    -outpath='UNETR_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=100 \

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='6' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MEN' -task='998' \
    -outpath='UNETR_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=50 \

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='7' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MEN' -task='998' \
    -outpath='UNETR_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=25 \

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='6' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MEN' -task='998' \
    -outpath='UNETR_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=10 \


# train from scratch Unimiss
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='5' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' -task='998' \
    -outpath='unimiss_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=100

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='6' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' -task='998' \
    -outpath='unimiss_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=50 

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='7' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' -task='998' \
    -outpath='unimiss_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=25

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' -task='998' \
    -outpath='unimiss_tf' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=10





# fintune
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MEN' \
    -task='998' -outpath='UNETR_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/MAE-pytorch/output/pretrain_mae_base_patch16_128/checkpoint-1999.pth' -split=100

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='1' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MEN' \
    -task='998' -outpath='UNETR_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/MAE-pytorch/output/pretrain_mae_base_patch16_128/checkpoint-1999.pth' -split=20

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='2' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MEN' \
    -task='998' -outpath='UNETR_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/MAE-pytorch/output/pretrain_mae_base_patch16_128/checkpoint-1999.pth' -split=10

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='3' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_UNETR_MEN' \
    -task='998' -outpath='UNETR_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/MAE-pytorch/output/pretrain_mae_base_patch16_128/checkpoint-1999.pth' -split=5


# fintune unimiss
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='1' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' \
    -task='998' -outpath='unimiss_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/UniMiSS/checkpoint.pth' -split=100

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='2' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' \
    -task='998' -outpath='unimiss_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/UniMiSS/checkpoint.pth' -split=50

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='3' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' \
    -task='998' -outpath='unimiss_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/UniMiSS/checkpoint.pth' -split=25

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='4' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' \
    -task='998' -outpath='unimiss_ft' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/UniMiSS/checkpoint.pth' -split=10


# inference
CUDA_VISIBLE_DEVICES=4 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_0" -t 998 -tr TrainerV2_Brats23_UNETR_MEN -chk model_best --overwrite_existing -split=100

CUDA_VISIBLE_DEVICES=5 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_1" -t 998 -tr TrainerV2_Brats23_UNETR_MEN -chk model_best --overwrite_existing -split=20

CUDA_VISIBLE_DEVICES=6 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_2" -t 998 -tr TrainerV2_Brats23_UNETR_MEN -chk model_best --overwrite_existing -split=10

CUDA_VISIBLE_DEVICES=7 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_3" -t 998 -tr TrainerV2_Brats23_UNETR_MEN -chk model_best --overwrite_existing -split=5


# fine-tune 10.15
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='only3D' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/only3D/checkpoint.pth' -split=100

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='1' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='only3D' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/only3D/checkpoint.pth' -split=50

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='2' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='only3D' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/only3D/checkpoint.pth' -split=25

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='3' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET' -task='997' \
    -outpath='only3D' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/only3D/checkpoint.pth' -split=10


MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='4' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' \
    -task='998' -outpath='only3D' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/only3D/checkpoint.pth' -split=100

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='5' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' \
    -task='998' -outpath='only3D' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/only3D/checkpoint.pth' -split=50

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='6' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' \
    -task='998' -outpath='only3D' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/only3D/checkpoint.pth' -split=25

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='7' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN' \
    -task='998' -outpath='only3D' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/only3D/checkpoint.pth' -split=10



CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_0" -t 998 -tr TrainerV2_Brats23_MEN -chk model_best --overwrite_existing -split=100

CUDA_VISIBLE_DEVICES=1 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_1" -t 998 -tr TrainerV2_Brats23_MEN -chk model_best --overwrite_existing -split=50

CUDA_VISIBLE_DEVICES=3 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_2" -t 998 -tr TrainerV2_Brats23_MEN -chk model_best --overwrite_existing -split=25

CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_3" -t 998 -tr TrainerV2_Brats23_MEN -chk model_best --overwrite_existing -split=10



CUDA_VISIBLE_DEVICES=4 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_4" -t 998 -tr TrainerV2_Brats23_MEN -chk model_best --overwrite_existing -split=100

CUDA_VISIBLE_DEVICES=5 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_5" -t 998 -tr TrainerV2_Brats23_MEN -chk model_best --overwrite_existing -split=50

CUDA_VISIBLE_DEVICES=6 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_6" -t 998 -tr TrainerV2_Brats23_MEN -chk model_best --overwrite_existing -split=25

CUDA_VISIBLE_DEVICES=7 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_7" -t 998 -tr TrainerV2_Brats23_MEN -chk model_best --overwrite_existing -split=10




CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_0" -t 997 -tr TrainerV2_Brats23_MET -chk model_best --overwrite_existing -split=100

CUDA_VISIBLE_DEVICES=1 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_1" -t 997 -tr TrainerV2_Brats23_MET -chk model_best --overwrite_existing -split=50

CUDA_VISIBLE_DEVICES=2 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_2" -t 997 -tr TrainerV2_Brats23_MET -chk model_best --overwrite_existing -split=25

CUDA_VISIBLE_DEVICES=3 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_3" -t 997 -tr TrainerV2_Brats23_MET -chk model_best --overwrite_existing -split=10

    

CUDA_VISIBLE_DEVICES=4 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_4" -t 997 -tr TrainerV2_Brats23_MET -chk model_best --overwrite_existing -split=100

CUDA_VISIBLE_DEVICES=5 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_5" -t 997 -tr TrainerV2_Brats23_MET -chk model_best --overwrite_existing -split=50

CUDA_VISIBLE_DEVICES=6 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_6" -t 997 -tr TrainerV2_Brats23_MET -chk model_best --overwrite_existing -split=25

CUDA_VISIBLE_DEVICES=7 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_7" -t 997 -tr TrainerV2_Brats23_MET -chk model_best --overwrite_existing -split=10