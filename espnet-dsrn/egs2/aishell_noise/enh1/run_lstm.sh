#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k


ngpu=1
device=3


stage=6
stop_stage=6

#$ -N train
#$ -cwd
#$ -j y


train_set=train
valid_set=dev
test_sets="test/clean test_seen/snr-10 test_seen/snr-5 test_seen/snr0 test_seen/snr5 test_snr_random"


enh_exp=exp/lstm/gpu03_3_lr0.001_SA_magnitude
#"$HOSTNAME"_"${device}"_lr"0.001"_SA
#gpu04_1_lr0.001_SA
#"$HOSTNAME"_"${device}"_lr"0.001"_SA

enh_config=conf/tuning/train_enh_lstm.yaml 

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
    --inference_model "valid.loss.best.pth" \
    --inference_tag "lstm"

