filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')

# test
python3 eval/new_classifier.py \
    --model linclr --net c3d --dataset ucf101-10clip \
    --seq_len 16 \
    --batch_size 8 \
    --temporal_ten_clip  --num_seq 10  -j 8  --gpu $1 \
    --ds 2 --aug_crop --rand_flip \
    --test  log/paper_table2_re_simclr_c3d/ft/${exp_name}/ucf/model/epoch99.pth.tar

