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





enh_asr_config=train_enh_asr_lstm_concate_fbank_transformer
inference_config=conf/decode_asr_transformer.yaml

enh_asr_exp=exp/enh_lstm_SA_concate_fbank/"${enh_asr_config}"_"$HOSTNAME"_"${device}"
#"${enh_asr_config}"_"$HOSTNAME"_"${device}"_lstm_300loss
#"${enh_asr_config}"_"$HOSTNAME"_"${device}""_lstm_300loss
#"${enh_asr_config}"_"$HOSTNAME"_"${device}"_lstm_50loss
#enh_asr_"${enh_asr_config}"_"$HOSTNAME"_"${device}"_lstm_15loss

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
    --input_size 160

