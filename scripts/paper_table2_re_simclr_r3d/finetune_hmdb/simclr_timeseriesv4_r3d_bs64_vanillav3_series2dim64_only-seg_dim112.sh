filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=15333 \
  eval/new_classifier.py \
    --prefix paper_table2_re_simclr_r3d --name_prefix ${exp_name} \
    --net r3d  --model linclr \
    --dataset hmdb51 --which_split 1 \
    --train_what ft \
    --seq_len 16 --num_seq 1 \
    --epochs 100 --schedule 30 60 80  --optim sgd  --img_dim 112 --img_resize_dim 128 --aug_crop --rand_flip --with_color_jitter \
    -j 4  \
    --lr 0.05 --wd 0.001 \
    --batch_size 4 \
    --print_freq 100 \
    --eval_freq 1 --save_freq 5 \
    --ds 2 \
    --pretrain log/paper_table2_re_simclr_r3d/pretrain/${exp_name}/model/epoch189.pth.tar


