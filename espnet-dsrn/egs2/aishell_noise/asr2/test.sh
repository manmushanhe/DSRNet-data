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
test_sets="test_seen/snr-10 test_seen/snr-5 test_seen/snr0 test_seen/snr5 test_snr_random"
# "test_seen/snr-10 test_seen/snr-5 test_seen/snr0 test_seen/snr5"
# "test_unseen/Babble/snr-10 test_unseen/Babble/snr-5 test_unseen/Babble/snr0 test_unseen/Babble/snr5"
# "Pink/snr-10 Pink/snr-5 Pink/snr0 Pink/snr5"

token_type=char

inference_asr_model=valid.acc.ave_10best.pth

# train related
asr_exp=exp/train_asr_transformer_specaug_notime_gpu03_0_lr0.001_4en_4de_fre0-10_notimemask

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
    --inference_config conf/decode_asr_transformer.yaml \
    --python  "${python}"  \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --inference_asr_model  "${inference_asr_model}"