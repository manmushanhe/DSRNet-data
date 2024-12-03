#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


ngpu=1
device=0

sample_rate=16k
#$ -N train
#$ -cwd
#$ -j y


stage=6
stop_stage=6


train_set=train
valid_set=dev
test_sets=test



enh_config="conf/tuning/train_enh_lstm_nawrn_one_branch.yaml"



enh_exp=exp/enh_nawrn_ratio_one_branch_pre_noise/gpu04_1_lr0.001_lstm
#gpu04_0_lr0.001_lstm
#"$HOSTNAME"_"${device}"_lr"0.001"_lstm
#"$HOSTNAME"_"${device}"_lr"0.001"_lstm_magnitude_300loss_enh_100loss_nawrn_SA_drop0.2_4en_de_max80
#enh_asr_exp=exp/enh_nawrn/train_enh_asr_lstm_nawrn_nobn_norelu_gpu05_2_lr0.001_lstm_magnitude_15loss_enh_15loss_nawrn_SA
python=/Work/python


./enh.sh \
    --stage "${stage}"   \
    --stop_stage "${stop_stage}"  \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --fs ${sample_rate} \
    --ngpu "${ngpu}" \
    --enh_exp "${enh_exp}"   \
    --device "${device}"    \
    --spk_num 1 \
    --enh_config "${enh_config}" \
    --python "${python}"   \
    --inference_model "valid.loss.best.pth" 
