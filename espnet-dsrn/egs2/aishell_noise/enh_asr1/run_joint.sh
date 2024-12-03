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





enh_asr_config=train_enh_asr_lstm_fbank_transformer
inference_config=conf/decode_asr_transformer.yaml


enh_asr_exp=exp/enh_nawrn/enh_asr_"${enh_asr_config}"_"$HOSTNAME"_"${device}"_lr"0.001"_lstm_nawrn_nobn_norelu_finetune
pretrained_model="/Work21/2021/luhaoyu/espnet/egs2/aishell_noise/asr1/exp/asr_train_train_asr_transformer_char_bins2000000_gpu04_1_lr0.0005/valid.acc.ave_10best.pth /Work21/2021/luhaoyu/espnet/egs2/aishell_noise/enh1/exp/lstm/gpu09_3_lr0.001_SA_nawrn_nobn_norelu/train.loss.ave_3best.pth"

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
    --test_sets "${test_sets}" \
    --pretrained_model "${pretrained_model}" 

