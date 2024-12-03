#!/usr/bin/env bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


ngpu=1
device=2 


#$ -N asr_decode
#$ -cwd
#$ -j y


stage=12
stop_stage=12

train_set=train
valid_set=dev
test_sets="test_seen/snr_5 test_seen/snr0 test_seen/snr5 test_seen/snr10 test_seen/snr15 test_seen/snr20"
#"test_seen_random test_unseen_random"
# test_unseen/snr_2.5 test_unseen/snr2.5 test_unseen/snr7.5 test_unseen/snr12.5 test_unseen/snr17.5 test_unseen/snr22.5 test_seen/snr_5 test_seen/snr0 test_seen/snr5 test_seen/snr10 test_seen/snr15 test_seen/snr20



inference_config=conf/decode_asr_transformer.yaml
inference_enh_asr_model=valid.acc.ave_10best.pth
#valid.acc.ave_10best.pth
#valid.acc.best.pth


enh_asr_exp=exp/enh_nawrn/train_enh_asr_lstm_nawrn_nobn_norelu_adaptive_weight_sub_band_fre_att_gpu03_3_lr0.001_lstm_magnitude_50loss_enh_50loss_nawrn_SA_adaptive_weight_sub_band_att
#train_enh_asr_lstm_nawrn_nobn_norelu_gpu05_2_lr0.001_lstm_magnitude_15loss_enh_15loss_nawrn_SA
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
