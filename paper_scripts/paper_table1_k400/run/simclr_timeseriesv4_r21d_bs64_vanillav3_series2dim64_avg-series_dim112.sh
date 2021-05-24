#!/bin/sh

filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')


echo 'start training'
pretrain
bash paper_scripts/paper_table1_k400/pretrain/${exp_name}.sh 0

# finetune
bash paper_scripts/paper_table1_k400/finetune/${exp_name}.sh 0
# test
bash paper_scripts/paper_table1_k400/test/${exp_name}.sh 0


# finetune hmdb
bash paper_scripts/paper_table1_k400/finetune_hmdb/${exp_name}.sh 0
# test hmdb
bash paper_scripts/paper_table1_k400/test_hmdb/${exp_name}.sh 0

 test_retrieval
 bash paper_scripts/paper_table1_k400/test_retrieval/${exp_name}.sh 0
