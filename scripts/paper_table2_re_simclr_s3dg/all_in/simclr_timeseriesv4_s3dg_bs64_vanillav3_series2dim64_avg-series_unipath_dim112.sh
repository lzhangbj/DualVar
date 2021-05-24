filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')

# pretrain
bash paper_scripts/paper_table2_re_simclr_s3dg/pretrain/${exp_name}.sh $1
# finetune
bash paper_scripts/paper_table2_re_simclr_s3dg/finetune/${exp_name}.sh $1
# test
bash paper_scripts/paper_table2_re_simclr_s3dg/test/${exp_name}.sh $1
# test_retrieval
bash paper_scripts/paper_table2_re_simclr_s3dg/test_retrieval/${exp_name}.sh $1