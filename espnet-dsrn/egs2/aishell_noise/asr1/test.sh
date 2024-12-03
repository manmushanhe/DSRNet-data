#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



ngpu=1
device=1

stage=12
stop_stage=13


#$ -N test
#$ -cwd
#$ -j y




train_set=train
valid_set=dev
test_sets="test_unseen/snr_2.5 test_unseen/snr2.5 test_unseen/snr7.5 test_unseen/snr12.5 test_unseen/snr17.5 test_unseen/snr22.5 test_seen/snr_5 test_seen/snr0 test_seen/snr5 test_seen/snr10 test_seen/snr15 test_seen/snr20"


token_type=char


# train related
asr_config=train_asr_transformer
asr_exp=exp/asr_train_train_asr_transformer_char_bins2000000_gpu06_1_lr0.001


python=/Work/python



./asr.sh \
    --stage "${stage}"   \
    --stop_stage "${stop_stage}"  \
    --ngpu "${ngpu}"   \
    --device "${device}"    \
    --use_lm false \
    --token_type char \
    --feats_type raw \
    --audio_format wav \
    --asr_exp "${asr_exp}"   \
    --asr_config conf/"${asr_config}".yaml \
    --inference_config conf/decode_asr_transformer.yaml \
    --python  "${python}"  \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}"