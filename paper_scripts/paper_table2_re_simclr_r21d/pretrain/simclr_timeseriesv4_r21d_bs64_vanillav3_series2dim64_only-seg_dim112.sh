filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')


python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=16000 \
  pretrain.py \
  --prefix paper_table2_re_simclr_r21d --name_prefix ${exp_name} \
  --model simclr_timeseriesv4 --aug_series --series_mode clip-sr --n_series 2 --series_dim 64 \
  --net r21d \
  --moco-t 0.07 \
  --dataset ucf101-2clip-stage-prototype --ds 4 -j 8 \
  --seq_len 16  --num_seq 3  --img_dim 112 \
  --schedule 120 160  --start_epoch 0  --epochs 200 \
  --batch_size 8 --lr 0.003  --wd 1e-4 --optim sgd \
  --print_freq 50 --eval_freq 5 --save_freq 5 \
  --aug_temp_consist --rand_flip