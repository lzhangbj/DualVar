filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')

# nearest neighbor
python3 eval/new_classifier.py \
    --model linclr --net s3dg --dataset ucf101 \
    --seq_len 16 \
    --batch_size 8 \
    --num_seq 10  -j 8  --gpu $1 --aug_crop --rand_flip \
    --retrieval \
    --ds 4 \
    --test log/paper_table2_moco_s3dg/pretrain/${exp_name}/model/epoch189.pth.tar


