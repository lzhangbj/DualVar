filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')

# finetune
bash paper_scripts/paper_table2_re_simclr_r21d/finetune/${exp_name}.sh $1
# test
bash paper_scripts/paper_table2_re_simclr_r21d/test/${exp_name}.sh $1
# test_retrieval
bash paper_scripts/paper_table2_re_simclr_r21d/test_retrieval/${exp_name}.sh $1