# train from scratch Unimiss
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='4' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN_DiT' -task='998' \
    -outpath='DiT_tf_200' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=100

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='5' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN_DiT' -task='998' \
    -outpath='DiT_tf_200' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=50 

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='6' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN_DiT' -task='998' \
    -outpath='DiT_tf_200' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=25

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='7' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN_DiT' -task='998' \
    -outpath='DiT_tf_200' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=10

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN_DiT' \
    -task='998' -outpath='DiT_ft_200_pre1' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/diffusion_preprocess_window/checkpoint.pth' -split=100

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='1' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN_DiT' \
    -task='998' -outpath='DiT_ft_200_pre1' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/diffusion_preprocess_window/checkpoint.pth' -split=50

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='2' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN_DiT' \
    -task='998' -outpath='DiT_ft_200_pre1' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/diffusion_preprocess_window/checkpoint.pth' -split=25

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='3' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN_DiT' \
    -task='998' -outpath='DiT_ft_200_pre1' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/diffusion_preprocess_window/checkpoint.pth' -split=10

# train from scratch UniMiss
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET_DiT' -task='997' \
    -outpath='DiT_tf_200' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=100

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='1' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET_DiT' -task='997' \
    -outpath='DiT_tf_200' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=50

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='2' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET_DiT' -task='997' \
    -outpath='DiT_tf_200' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=25

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='3' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET_DiT' -task='997' \
    -outpath='DiT_tf_200' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=10


MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='4' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET_DiT' -task='997' \
    -outpath='DiT_ft_200_pre1' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/diffusion_preprocess_window/checkpoint.pth' -split=100

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='5' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET_DiT' -task='997' \
    -outpath='DiT_ft_200_pre1' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/diffusion_preprocess_window/checkpoint.pth' -split=50

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='6' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET_DiT' -task='997'\
    -outpath='DiT_ft_200_pre1' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/diffusion_preprocess_window/checkpoint.pth' -split=25

MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='7' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MET_DiT' -task='997' \
    -outpath='DiT_ft_200_pre1' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='/data4/kangluoyao/UniMiSS-code/UniMiSS/snapshots/diffusion_preprocess_window/checkpoint.pth' -split=10 



## inference
CUDA_VISIBLE_DEVICES=0 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_0" -t 997 -tr TrainerV2_Brats23_MET_DiT -chk model_best --overwrite_existing -split=100

CUDA_VISIBLE_DEVICES=1 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_1" -t 997 -tr TrainerV2_Brats23_MET_DiT -chk model_best --overwrite_existing -split=50

CUDA_VISIBLE_DEVICES=2 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_2" -t 997 -tr TrainerV2_Brats23_MET_DiT -chk model_best --overwrite_existing -split=25

CUDA_VISIBLE_DEVICES=3 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task997_BraTS2023_MET/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_3" -t 997 -tr TrainerV2_Brats23_MET_DiT -chk model_best --overwrite_existing -split=10

    

CUDA_VISIBLE_DEVICES=4 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_4" -t 998 -tr TrainerV2_Brats23_MEN_DiT -chk model_best --overwrite_existing -split=100

CUDA_VISIBLE_DEVICES=5 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_5" -t 998 -tr TrainerV2_Brats23_MEN_DiT -chk model_best --overwrite_existing -split=50

CUDA_VISIBLE_DEVICES=6 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_6" -t 998 -tr TrainerV2_Brats23_MEN_DiT -chk model_best --overwrite_existing -split=25

CUDA_VISIBLE_DEVICES=7 python predict_simple.py -i "/data2/public_data/ASA/UniMiss/preprocessed/BCV/preprocess/nnUNet_raw/nnUNet_raw_data/Task998_BraTS2023_MEN/imagesTs" \
    -o "/data4/kangluoyao/UniMiSS-code/UniMiSS/Downstream/brats_tmp/brats2021_7" -t 998 -tr TrainerV2_Brats23_MEN_DiT -chk model_best --overwrite_existing -split=10