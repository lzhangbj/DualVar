filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')

# nearest neighbor
python3 classifier.py \
    --model linclr --net r21d --dataset ucf101 \
    --seq_len 16 \
    --batch_size 8 \
    --num_seq 10  -j 8  --gpu $1 --aug_crop --rand_flip \
    --retrieval \
    --ds 4 \
    --test log/paper_table1_k400/pretrain/${exp_name}/model/epoch189.pth.tar


