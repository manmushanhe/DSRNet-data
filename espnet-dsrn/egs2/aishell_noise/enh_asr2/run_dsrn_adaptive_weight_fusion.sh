#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


ngpu=1
device=2


#$ -N train
#$ -cwd
#$ -j y


stage=11
stop_stage=11


train_set=train
valid_set=dev
test_sets=test



enh_asr_config=train_enh_asr_dsrn_adaptive_weight_fusion
inference_config=conf/decode_asr_transformer.yaml


enh_asr_exp=exp/enh_dsrn_adaptive_weight_fusion/"$HOSTNAME"_"${device}"_lr"0.001"_300loss_enh_100loss_dsrn
#gpu03_2_lr0.001_300loss_enh_100loss_nawrn_4en_de
#"$HOSTNAME"_"${device}"_lr"0.001"_300loss_enh_100loss_nawrn_4en_de
#"$HOSTNAME"_"${device}"_lr"0.001"_lstm_magnitude_300loss_enh_100loss_nawrn_SA_drop0.2_4en_de_max80
#enh_asr_exp=exp/enh_nawrn/train_enh_asr_lstm_nawrn_nobn_norelu_gpu05_2_lr0.001_lstm_magnitude_15loss_enh_15loss_nawrn_SA
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
    --test_sets "${test_sets}"  \
    --input_size 80

