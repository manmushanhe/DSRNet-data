#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


ngpu=1
device=0


#$ -N train
#$ -cwd
#$ -j y


stage=11
stop_stage=11


train_set=train
valid_set=dev
test_sets=test





enh_asr_config=train_enh_asr_lstm_nawrn_nobn_norelu_adaptive_weight
inference_config=conf/decode_asr_transformer.yaml


enh_asr_exp=exp/enh_nawrn_new/"${enh_asr_config}"_"$HOSTNAME"_"${device}"_lr"0.001"_lstm_magnitude_200loss_enh_200loss_nawrn_SA_adaptive
#enh_asr_exp=exp/enh_nawrn/train_enh_asr_lstm_nawrn_nobn_norelu_gpu05_2_lr0.001_lstm_magnitude_15loss_enh_15loss_nawrn_SA
python=/Work21/2021/luhaoyu/espnet/tools/anaconda/envs/espnet/bin/python



./enh_asr.sh \
    --spk_num 1 \
    --stage "${stage}"   \
    --stop_stage "${stop_stage}"  \
    --token_type char \
    --feats_type raw \
    --audio_format wav  \
    --ngpu "${ngpu}"   \
    --device "${device}"    \
    --enh_asr_exp "${enh_asr_exp}"   \
    --enh_asr_config conf/"${enh_asr_config}".yaml \
    --inference_config "${inference_config}" \
    --python "${python}"   \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" 

