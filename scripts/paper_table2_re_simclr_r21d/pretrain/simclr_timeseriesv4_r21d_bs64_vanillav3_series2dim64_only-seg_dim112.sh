filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')


# NCCL_DEBUG=INFO
#OMP_NUM_THREADS=8 NCCL_IB_DISABLE=0 NCCL_IB_GID_INDEX=3 NCCL_IB_HCA=mlx5_2 NCCL_SOCKET_IFNAME=eth0 \
#python3 -m torch.distributed.launch --nproc_per_node=8 \
#  --nnodes=${ARNOLD_WORKER_NUM} --node_rank=${ARNOLD_ID} --master_addr=${METIS_WORKER_0_HOST} --master_port=${METIS_WORKER_0_PORT} \
python3 -m torch.distributed.launch --nproc_per_node=8 \
  new_vid_frame_exp.py \
  --prefix paper_table2_re_simclr_r21d --name_prefix ${exp_name} \
  --model simclr_timeseriesv4 --aug_same_series --n_series 2 --series_dim 64 --series_T 0.5 \
  --net r21d --series_method v3 \
  --moco-t 0.07 \
  --ft_mode \
  --dataset ucf101-2clip-stage-prototype --ds 4 -j 8 \
  --seq_len 16  --num_seq 3  --img_dim 112 --img_resize_dim 128 \
  --schedule 120 160  --start_epoch 0  --epochs 200 \
  --batch_size 8 --lr 0.003  --wd 1e-4 --optim sgd \
  --print_freq 50 --eval_freq 5 --save_freq 5 \
  --n_block 1 --aug_color_jitter --aug_gaussian_blur --aug_temp_consist --rand_flip --aug_crop --transform_1