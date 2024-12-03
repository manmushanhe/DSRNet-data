#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


ngpu=1
device=3


#$ -N train
#$ -cwd
#$ -j y


stage=11
stop_stage=11


train_set=train
valid_set=dev
test_sets=test





enh_asr_config=train_enh_asr_lstm_fbank_pretrain_freeze
inference_config=conf/decode_asr_transformer.yaml

enh_asr_exp=exp/enh_lstm_pretrain/train_enh_asr_lstm_fbank_pretrain_freeze_gpu03_3_lr0.001_lstm_magnitude_pretrain_freeze
#"${enh_asr_config}"_"$HOSTNAME"_"${device}"_lr"0.001"_lstm_magnitude_300loss_enh_4en_4de_max70
#"${enh_asr_config}"_"$HOSTNAME"_"${device}"_lr"0.001"_lstm_magnitude_50loss_enh_4en_4de
#enh_asr_"${enh_asr_config}"_"$HOSTNAME"_"${device}"_lr"0.001"_lstm_magnitude_15loss_enh_4en_4de
python=/Work/python



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
    --test_sets "${test_sets}" \
    --pretrained_model "/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh1/exp/lstm/gpu03_3_lr0.001_SA_magnitude/pretrain_enh_model.pth"    \
    --freeze_param  "enh_model"  \
    --input_size 80

