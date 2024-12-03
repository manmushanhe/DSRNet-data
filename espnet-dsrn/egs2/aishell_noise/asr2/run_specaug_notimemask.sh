#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail



ngpu=1
device=0

stage=11
stop_stage=11


#$ -N train
#$ -cwd
#$ -j y




train_set=train
valid_set=dev
test_sets="test_seen_random test_unseen_random"


token_type=char

# train related
asr_config=train_asr_transformer_specaug_notime
asr_exp=exp/"${asr_config}"_"$HOSTNAME"_"${device}"_lr"0.001"_4en_4de_fre0-10_notimemask
#"${asr_config}"_"$HOSTNAME"_"${device}"_lr"0.0005"_4en_4de_fre0-10_time0-10_num1
#train_asr_transformer_specaug_char_gpu03_1_lr0.0005_4en_4de_fre0-30_num_2
#"${asr_config}"_"$HOSTNAME"_"${device}"_lr"0.002"_4en_4de_fre0-30_num2
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