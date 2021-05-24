#!/bin/sh

#
#filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
#exp_name=$(echo "$filename" | cut -f 1 -d '.')
#
#cd /opt/tiger/coclr
#
#echo 'start training'
## pretrain
#bash paper_scripts/paper_table1_k400/pretrain/${exp_name}.sh 0
#
#if [ ${ARNOLD_ID} -eq 0 ]; then
#  echo "upoading pretraining data"
#  hdfs dfs -put log/paper_table1_k400 /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/
#
#  # finetune
#  bash paper_scripts/paper_table1_k400/finetune/${exp_name}.sh 0
#  echo "upoading finetuning data"
#  hdfs dfs -put log/paper_table1_k400 /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/
#  # test
#  bash paper_scripts/paper_table1_k400/test/${exp_name}.sh 0
#
#
#  # finetune hmdb
#  bash paper_scripts/paper_table1_k400/finetune_hmdb/${exp_name}.sh 0
#  echo "upoading finetuning data"
#  hdfs dfs -put log/paper_table1_k400 /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/
#  # test hmdb
#  bash paper_scripts/paper_table1_k400/test_hmdb/${exp_name}.sh 0
#
#  # test_retrieval
#  bash paper_scripts/paper_table1_k400/test_retrieval/${exp_name}.sh 0
#
#  echo "upoading testing data"
#  hdfs dfs -put log/paper_table1_k400 /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/
#
#fi

filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')

cd /opt/tiger/coclr

bash prepare_hdfs_data.sh

mkdir log
hdfs dfs -get /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/paper_table1_k400 ./log/

# echo 'start training'
# pretrain
# bash paper_scripts/paper_table1_k400/pretrain/${exp_name}.sh 0

# if [ ${ARNOLD_ID} -eq 0 ]; then
# echo "upoading pretraining data"
# hdfs dfs -put log/paper_table1_k400 /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/

# finetune
bash paper_scripts/paper_table1_k400/finetune/${exp_name}.sh 0
# echo "upoading finetuning data"
# hdfs dfs -put log/paper_table1_k400 /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/
# test
bash paper_scripts/paper_table1_k400/test/${exp_name}.sh 0


# finetune hmdb
bash paper_scripts/paper_table1_k400/finetune_hmdb/${exp_name}.sh 0
# echo "upoading finetuning data"
# hdfs dfs -put log/paper_table1_k400 /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/
# test hmdb
bash paper_scripts/paper_table1_k400/test_hmdb/${exp_name}.sh 0

# test_retrieval
# bash paper_scripts/paper_table1_k400/test_retrieval/${exp_name}.sh 0

# echo "upoading testing data"
# hdfs dfs -put log/paper_table1_k400 /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/
# fi

