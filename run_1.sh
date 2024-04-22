CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7' \
python -m torch.distributed.launch --nproc_per_node=8 --master_port=$RANDOM main_3D_DiT_latent.py \
--interval=16 \
--arch='model_small' \
--data_path='/data2/public_data/ASA/UniMiss/brats2023/' \
--list_path2D='2D_images.txt' \
--list_path3D='3D_images.txt' \
--batch_size_per_gpu=6 \
--epochs=1000 \
--lr=0.0001 \
--num_workers=4 \
--momentum_teacher=0.996 \
--clip_grad=0.3 \
--output_dir='snapshots/DiT_nomod/' \
