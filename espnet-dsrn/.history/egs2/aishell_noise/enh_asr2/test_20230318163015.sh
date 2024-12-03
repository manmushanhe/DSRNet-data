#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


ngpu=0
device=2 


#$ -N asr_decode
#$ -cwd
#$ -j y


stage=12
stop_stage=13


train_set=train
valid_set=dev
test_sets="test_seen/snr-10 test_seen/snr-5 test_seen/snr0 test_seen/snr5"
#"test_seen/snr-10 test_seen/snr-5 "
# "test_snr_random"
# "White/random" "Volvo/random" "F16/random" "Factory1/random"
# "test_spec_snr-10"
# "test_spec_seen"
# "test_seen/snr-10 test_seen/snr-5 test_seen/snr0 test_seen/snr5"
# "test_unseen/Babble/snr-10 test_unseen/Babble/snr-5 test_unseen/Babble/snr0 test_unseen/Babble/snr5"
# "Pink/snr-10 Pink/snr-5 Pink/snr0 Pink/snr5"


inference_config=conf/decode_asr_transformer.yaml
inference_enh_asr_model=valid.acc.ave_10best.pth
#valid.acc.ave_10best.pth
#valid.acc.best.pth


enh_asr_exp=exp/enh_dsrn_conv2d_adaptive_weight/gpu09_2_lr0.001_64ch_3cnn_64relu_no_residual


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
    --inference_config "${inference_config}" \
    --python "${python}"   \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}"  \
    --inference_enh_asr_model  "${inference_enh_asr_model}"
