#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


ngpu=1
device=1


#$ -N train
#$ -cwd
#$ -j y


stage=11
stop_stage=11


train_set=train
valid_set=dev
test_sets=test





enh_asr_config=train_enh_asr_lstm_cascaded_transformer
inference_config=conf/decode_asr_transformer.yaml

enh_asr_exp=exp/enh_lstm_cascaded/enh_asr_train_enh_asr_lstm_cascaded_transformer_gpu04_1_lr0.0005_lstm_magnitude_300loss_enh_4en_4de
#enh_asr_"${enh_asr_config}"_"$HOSTNAME"_"${device}"_lr"0.0005"_lstm_magnitude_300loss_enh_4en_4de
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
    --input_size 80

