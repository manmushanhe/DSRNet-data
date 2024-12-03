#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

sample_rate=16k
# Path to a directory containing extra annotations for CHiME4
# Run `local/data.sh` for more information.
extra_annotations=


ngpu=1
device=1

#$ -N train
#$ -cwd
#$ -j y


stage=8
stop_stage=8


train_set=train
valid_set=dev
test_sets=test

enh_config=train_enh_lstm
batch_bins=$(cat conf/tuning/train_enh_lstm.yaml  | grep "batch_bins" | awk '{print $2}')

enh_exp=exp/blstm/"${enh_config}"_"$HOSTNAME"_"${device}"_lr"0.0001"_lstm_spectrogram_SA

python=/Work/python
./enh.sh \
    --stage "${stage}"   \
    --stop_stage "${stop_stage}"  \
    --fs ${sample_rate} \
    --ngpu "${ngpu}" \
    --spk_num 1 \
    --device "${device}"    \
    --enh_exp "${enh_exp}"   \
    --enh_config conf/tuning/"${enh_config}".yaml \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --python "${python}"  \
    --inference_model "front_baseline"