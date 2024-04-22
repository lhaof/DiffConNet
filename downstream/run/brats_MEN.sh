#DiT train_from_scratch
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN_DiT' -task='998' \
    -outpath='DiT' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -split=100

#DiT fine-tune
MKL_NUM_THREADS=1,NUMEXPR_NUM_THREADS=1,OMP_NUM_THREADS=1 python -u run_training.py --deterministic -gpu='0' -network='3d_fullres' -network_trainer='TrainerV2_Brats23_MEN_DiT' -task='998' \
    -outpath='DiT' -norm_cfg='IN' -activation_cfg='LeakyReLU' -epochs=200 -pre_train -pre_path='checkpoint.pth' -split=100

# DiT inference
CUDA_VISIBLE_DEVICES=7 python predict_simple.py -i "testset_path" \
    -o "output_path" -t 998 -tr TrainerV2_Brats23_MEN_DiT -chk model_best --overwrite_existing -split=100


