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





enh_asr_config=train_enh_asr_lstm_nawrn_nobn_norelu_ratio_one_branch_adaptive_weight_freeze
inference_config=conf/decode_asr_transformer.yaml

enh_asr_exp=exp/pretrainenh_nawr_one_branch_pre_noise_no_drop/"$HOSTNAME"_"${device}"_lr"0.001"_freeze_lr0.001_pretrain_freeze_noenhloss_nonawrloss
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
    --pretrained_model "/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh_asr2/exp/pretrainenh_nawr_one_branch_pre_noise_no_drop/pretrain_nawr_one_branch_pre_noise_no_drop.pth"    \
    --freeze_param  "enh_model"  \
    --input_size 80

