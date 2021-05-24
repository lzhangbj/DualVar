#!/bin/sh

filename="$(basename "$(test -L "$0" && readlink "$0" || echo "$0")")"
exp_name=$(echo "$filename" | cut -f 1 -d '.')

cd /opt/tiger/coclr

bash prepare_hdfs_data.sh

echo 'start training'
# pretrain
bash paper_scripts/paper_table2_re_simclr_s3dg/pretrain/${exp_name}.sh 0
echo "upoading pretraining data"
hdfs dfs -put log/paper_table2_re_simclr_s3dg /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/

# finetune
bash paper_scripts/paper_table2_re_simclr_s3dg/finetune/${exp_name}.sh 0
echo "upoading finetuning data"
hdfs dfs -put log/paper_table2_re_simclr_s3dg /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/
# test
bash paper_scripts/paper_table2_re_simclr_s3dg/test/${exp_name}.sh 0


# finetune hmdb
bash paper_scripts/paper_table2_re_simclr_s3dg/finetune_hmdb/${exp_name}.sh 0
echo "upoading finetuning data"
hdfs dfs -put log/paper_table2_re_simclr_s3dg /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/
# test hmdb
bash paper_scripts/paper_table2_re_simclr_s3dg/test_hmdb/${exp_name}.sh 0

# test_retrieval
bash paper_scripts/paper_table2_re_simclr_s3dg/test_retrieval/${exp_name}.sh 0

echo "upoading testing data"
hdfs dfs -put log/paper_table2_re_simclr_s3dg /home/byte_arnold_hl_vc/zhanglin99/CoCLR/log/
