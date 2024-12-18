#!/usr/bin/env bash

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}
min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}
SECONDS=0

# General configuration
stage=1              # Processes starts from the specified stage.
stop_stage=10000     # Processes is stopped at the specified stage.
skip_data_prep=false # Skip data preparation stages.
skip_train=false     # Skip training stages.
skip_eval=false      # Skip decoding and evaluation stages.

ngpu=1               # The number of gpus ("0" uses cpu, otherwise use gpu).
num_nodes=1          # The number of nodes.
nj=32                # The number of parallel jobs.
inference_nj=32      # The number of parallel jobs in decoding.
gpu_inference=false  # Whether to perform gpu decoding.
dumpdir=dump         # Directory to dump features.  堆放特征的目录
expdir=exp           # Directory to save experiments.
python=python3       # Specify python to execute espnet commands.
device=
freeze_param=

# Data preparation related
local_data_opts= # The options given to local/data.sh.



# Feature extraction related
feats_type=raw       # Feature type (raw or fbank_pitch).
audio_format=        # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw).
fs=16k               # Sampling rate.
min_wav_duration=0.1 # Minimum duration in second.
max_wav_duration=20  # Maximum duration in second.

# Tokenization related
token_type=char      # Tokenization type (char or bpe).

oov="<unk>"         # Out of vocabulary symbol.
blank="<blank>"     # CTC blank symbol
sos_eos="<sos/eos>" # sos and eos symbole




use_word_lm=false # Whether to use word language model.
num_splits_lm=1   # Number of splitting for lm corpus.
# shellcheck disable=SC2034
word_vocab_size=10000 # Size of word vocabulary.

# ASR model related
enh_asr_tag=       # Suffix to the result dir for asr model training.
enh_asr_exp=       # Specify the directory path for ASR experiment.
                   # If this option is specified, enh_asr_tag is ignored.
enh_asr_stats_dir= # Specify the directory path for ASR statistics.
enh_asr_config=    # Config for asr model training.
enh_asr_args=      # Arguments for asr model training, e.g., "--max_epoch 10".
                   # Note that it will overwrite args in asr config.
pretrained_model=          # Pretrained model to load
ignore_init_mismatch=true # Ignore initial mismatch
feats_normalize=none           # Normalizaton layer type.
num_splits_asr=1           # Number of splitting for lm corpus. lm语料库的分开数
input_size=


# Decoding related
use_k2=false      # Whether to use k2 based decoder
batch_size=1
inference_tag=    # Suffix to the result dir for decoding.
inference_config= # Config for decoding.
asr_inference_args= # Arguments for decoding, e.g., "--lm_weight 0.1".
                    # Note that it will overwrite args in inference config.
enh_inference_args="--normalize_output_wav true"     # 归一化输出音频，在enhance_asr中没有用到
inference_lm=valid.loss.ave.pth       # Language model path for decoding.
inference_enh_asr_model=                  # ASR model path for decoding.
                                          # e.g.
                                          # inference_enh_asr_model=train.loss.best.pth
                                          # inference_enh_asr_model=3epoch.pth
                                          # inference_enh_asr_model=valid.acc.best.pth
                                          # inference_enh_asr_model=valid.loss.ave.pth

# Enhancement related arguments
spk_num=1   # Number of speakers
noise_type_num=1
dereverb_ref_num=1
# Evaluation related
scoring_protocol="STOI SDR SAR SIR SI_SNR"
ref_channel=
inference_enh_tag=      # Prefix to the result dir for ENH inference.
inference_enh_config=   # Config for enhancement.

# Enh Training data related
use_dereverb_ref=false
use_noise_ref=true

# [Task dependent] Set the datadir name created by local/data.sh
train_set=       # Name of training set.
valid_set=       # Name of validation set used for monitoring/tuning network training.
test_sets=       # Names of test sets. Multiple items (e.g., both dev and eval sets) can be specified.

nlsyms_txt=none  # Non-linguistic symbol list if existing.
cleaner=none     # Text cleaner.
g2p=none         # g2p method (needed if token_type=phn).
lang=noinfo      # The language type of corpus.
score_opts=                # The options given to sclite scoring
local_score_opts=          # The options given to local/score.sh.



help_message=$(cat << EOF
Usage: $0 --train-set "<train_set_name>" --valid-set "<valid_set_name>" --test_sets "<test_set_names>"

Options:
    # General configuration
    --stage          # Processes starts from the specified stage (default="${stage}").
    --stop_stage     # Processes is stopped at the specified stage (default="${stop_stage}").
    --skip_data_prep # Skip data preparation stages (default="${skip_data_prep}").
    --skip_train     # Skip training stages (default="${skip_train}").
    --skip_eval      # Skip decoding and evaluation stages (default="${skip_eval}").
    --ngpu           # The number of gpus ("0" uses cpu, otherwise use gpu, default="${ngpu}").
    --num_nodes      # The number of nodes (default="${num_nodes}").
    --nj             # The number of parallel jobs (default="${nj}").
    --inference_nj   # The number of parallel jobs in decoding (default="${inference_nj}").
    --gpu_inference  # Whether to perform gpu decoding (default="${gpu_inference}").
    --dumpdir        # Directory to dump features (default="${dumpdir}").
    --expdir         # Directory to save experiments (default="${expdir}").
    --python         # Specify python to execute espnet commands (default="${python}").
    --device 
    --freeze_param

    # Data preparation related
    --local_data_opts # The options given to local/data.sh (default="${local_data_opts}").



    # Feature extraction related
    --feats_type       # Feature type (raw, fbank_pitch or extracted, default="${feats_type}").
    --audio_format     # Audio format: wav, flac, wav.ark, flac.ark  (only in feats_type=raw, default="${audio_format}").
    --fs               # Sampling rate (default="${fs}").
    --min_wav_duration # Minimum duration in second (default="${min_wav_duration}").
    --max_wav_duration # Maximum duration in second (default="${max_wav_duration}").

    # Tokenization related
    --token_type              # Tokenization type (char or bpe, default="${token_type}").
    --oov                     # Out of vocabulary symbol (default="${oov}").
    --blank                   # CTC blank symbol (default="${blank}").
    --sos_eos                 # sos and eos symbole (default="${sos_eos}").


    # ASR model related
    --input_size  
    --enh_asr_tag          # Suffix to the result dir for asr model training (default="${enh_asr_tag}").  asr模型训练的后缀
    --enh_asr_exp          # Specify the directory path for ASR experiment.
                       # If this option is specified, enh_asr_tag is ignored (default="${enh_asr_exp}").
    --enh_asr_stats_dir    # Specify the directory path for ASR statistics (default="${enh_asr_stats_dir}").
    --enh_asr_config       # Config for asr model training (default="${enh_asr_config}").
    --enh_asr_args         # Arguments for asr model training (default="${enh_asr_args}").
                           # e.g., --enh_asr_args "--max_epoch 10"
                           # Note that it will overwrite args in asr config.
    --pretrained_model          # Pretrained model to load (default="${pretrained_model}").
    --ignore_init_mismatch      # Ignore mismatch parameter init with pretrained model (default="${ignore_init_mismatch}").
    --feats_normalize  # Normalizaton layer type (default="${feats_normalize}").
    --num_splits_asr   # Number of splitting for lm corpus  (default="${num_splits_asr}").

    # Decoding related
    --inference_tag       # Suffix to the result dir for decoding (default="${inference_tag}").
    --inference_config    # Config for decoding (default="${inference_config}").
    --asr_inference_args      # Arguments for decoding (default="${asr_inference_args}").
                              # e.g., --asr_inference_args "--lm_weight 0.1"
                              # Note that it will overwrite args in inference config.
    --enh_inference_args      # Arguments for enhancement (default="${enh_inference_args}").
    --inference_lm        # Language model path for decoding (default="${inference_lm}").
    --inference_enh_asr_model # ASR model path for decoding (default="${inference_enh_asr_model}").


    --spk_num             # number of speakers
    --noise_type_num   # Number of noise types in the input audio (default="${noise_type_num}")
    --dereverb_ref_num # Number of references for dereverberation (default="${dereverb_ref_num}")
                        # 去混响参考数量
    --use_dereverb_ref # Whether or not to use dereverberated signal as an additional reference
                         for training a dereverberation model (default="${use_dereverb_ref}")
                         #使用去混响的信号作为额外参考
    --use_noise_ref    # Whether or not to use noise signal as an additional reference
                         for training a denoising model (default="${use_noise_ref}")
    # Enhancement Evaluation related
    --scoring_protocol    # Metrics to be used for scoring (default="${scoring_protocol}")
    --ref_channel         # Reference channel of the reference speech will be used if the model
                            output is single-channel and reference speech is multi-channel
                            (default="${ref_channel}")
    # [Task dependent] Set the datadir name created by local/data.sh
    --train_set     # Name of training set (required).
    --valid_set     # Name of validation set used for monitoring/tuning network training (required).
    --test_sets     # Names of test sets.
                    # Multiple items (e.g., both dev and eval sets) can be specified (required).


    --nlsyms_txt    # Non-linguistic symbol list if existing (default="${nlsyms_txt}").
    --cleaner       # Text cleaner (default="${cleaner}").
    --g2p           # g2p method (default="${g2p}").
    --lang          # The language type of corpus (default=${lang}).
    --score_opts             # The options given to sclite scoring (default="{score_opts}").
    --local_score_opts       # The options given to local/score.sh (default="{local_score_opts}").
EOF
)

log "$0 $*"
# Save command line args for logging (they will be lost after utils/parse_options.sh)
run_args=$(pyscripts/utils/print_args.py $0 "$@")
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    log "Error: No positional arguments are required."
    exit 2
fi

. ./path.sh
. ./cmd.sh


# Check required arguments
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${valid_set}" ] && { log "${help_message}"; log "Error: --valid_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };

[ ${spk_num} -gt 1 ] && { log "${help_message}"; log "Error: --spk_num only 1 is supported"; exit 2; };

# Check feature type
if [ "${feats_type}" = raw ]; then
    data_feats=${dumpdir}/raw
elif [ "${feats_type}" = fbank_pitch ]; then
    data_feats=${dumpdir}/fbank_pitch
elif [ "${feats_type}" = fbank ]; then
    data_feats=${dumpdir}/fbank
elif [ "${feats_type}" == extracted ]; then
    data_feats=${dumpdir}/extracted
else
    log "${help_message}"
    log "Error: not supported: --feats_type ${feats_type}"
    exit 2
fi

# Extra files for enhancement process
# 语音增强过程中额外的文件
utt_extra_files="utt2category text utt2lang"



# Check tokenization type
if [ "${lang}" != noinfo ]; then
    token_listdir=data/${lang}_token_list
else
    token_listdir=data/token_list
fi

chartoken_list="${token_listdir}"/char/tokens.txt
# NOTE: keep for future development.
# shellcheck disable=SC2034
wordtoken_list="${token_listdir}"/word/tokens.txt

if [ "${token_type}" = char ]; then
    token_list="${chartoken_list}"
    bpemodel=none
elif [ "${token_type}" = word ]; then
    token_list="${wordtoken_list}"
    bpemodel=none
else
    log "Error: not supported --token_type '${token_type}'"
    exit 2
fi



# Set tag for naming of model directory
if [ -z "${enh_asr_tag}" ]; then
    if [ -n "${enh_asr_config}" ]; then
        enh_asr_tag="$(basename "${enh_asr_config}" .yaml)_${feats_type}"
    else
        enh_asr_tag="train_${feats_type}"
    fi
    if [ "${lang}" != noinfo ]; then
        enh_asr_tag+="_${lang}_${token_type}"
    else
        enh_asr_tag+="_${token_type}"
    fi
    # Add overwritten arg's info
    if [ -n "${enh_asr_args}" ]; then
        enh_asr_tag+="$(echo "${enh_asr_args}" | sed -e "s/--/\_/g" -e "s/[ |=/]//g")"
    fi
fi


# The directory used for collect-stats mode
if [ -z "${enh_asr_stats_dir}" ]; then
    if [ "${lang}" != noinfo ]; then
        enh_asr_stats_dir="${expdir}/enh_asr_stats_${feats_type}_${lang}_${token_type}"
    else
        enh_asr_stats_dir="${expdir}/enh_asr_stats_${feats_type}_${token_type}"
    fi

fi

# The directory used for training commands
if [ -z "${enh_asr_exp}" ]; then
    enh_asr_exp="${expdir}/enh_asr_${enh_asr_tag}"
fi



if [ -z "${inference_tag}" ]; then
    if [ -n "${inference_config}" ]; then
        inference_tag="$(basename "${inference_config}" .yaml)"
    else
        inference_tag=inference
    fi
    # Add overwritten arg's info
    if [ -n "${asr_inference_args}" ]; then
        inference_tag+="$(echo "${asr_inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    if [ -n "${enh_inference_args}" ]; then
        inference_tag+="$(echo "${enh_inference_args}" | sed -e "s/--/\_/g" -e "s/[ |=]//g")"
    fi
    inference_tag+="_enh_asr_model_$(echo "${inference_enh_asr_model}" | sed -e "s/\//_/g" -e "s/\.[^.]*$//g")"

    if "${use_k2}"; then
      inference_tag+="_use_k2"
    fi
fi

if [ -z "${inference_enh_tag}" ]; then
    if [ -n "${inference_enh_config}" ]; then
        inference_enh_tag="$(basename "${inference_enh_config}" .yaml)"
    else
        inference_enh_tag=enhanced
    fi
fi

# ========================== Main stages start from here. ==========================

if ! "${skip_data_prep}"; then
    if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
        log "Stage 1: Data preparation for data/${train_set}, data/${valid_set}, etc."
        # [Task dependent] Need to create data.sh for new corpus
        local/data.sh ${local_data_opts}
    fi


    if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
        if [ "${feats_type}" = raw ]; then
            log "Stage 3: Format wav.scp: data/ -> ${data_feats}"

            # ====== Recreating "wav.scp" ======
            # 从新创建wav.scp
            # Kaldi-wav.scp, which can describe the file path with unix-pipe, like "cat /some/path |",
            # shouldn't be used in training process.
            # "format_wav_scp.sh" dumps such pipe-style-wav to real audio file
            # and it can also change the audio-format and sampling rate.
            # If nothing is need, then format_wav_scp.sh does nothing:
            # i.e. the input file format and rate is same as the output.

            #for dset in "${train_set}" "${valid_set}" ${test_sets}; do
            for dset in "${train_set}" "${valid_set}" ; do
                if [ "${dset}" = "${train_set}" ] || [ "${dset}" = "${valid_set}" ]; then
                    _suf="/org"
                else
                    _suf=""
                fi
                utils/copy_data_dir.sh --validate_opts --non-print data/"${dset}" "${data_feats}${_suf}/${dset}"

                # expand the utt_extra_files for multi-references
                expand_utt_extra_files=""
                # utt_extra_files "utt2category text utt2lang" 
                for extra_file in ${utt_extra_files}; do
                    # with regex to suuport multi-references
                    for single_file in "data/${dset}/${extra_file}"*; do
                        # 如果单个文件不存在，则进行下一轮循环
                        if [ ! -f "${single_file}" ]; then
                            continue
                        fi
                        # 如果存在则复制到堆放特征的目录下
                        cp ${single_file} "${data_feats}${_suf}/${dset}"
                        expand_utt_extra_files="${expand_utt_extra_files} $(basename ${single_file})"
                    done
                done
                echo "${expand_utt_extra_files}"

                rm -f ${data_feats}${_suf}/${dset}/{segments,wav.scp,reco2file_and_channel,reco2dur}
                _opts=
                if [ -e data/"${dset}"/segments ]; then
                    # "segments" is used for splitting wav files which are written in "wav".scp
                    # into utterances. The file format of segments:
                    #   <segment_id> <record_id> <start_time> <end_time>
                    #   "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5"
                    # Where the time is written in seconds.
                    _opts+="--segments data/${dset}/segments "
                fi

                _spk_list=" "
                #for i in $(seq ${spk_num}); do
                    #_spk_list+="spk${i} "
                #done
                if $use_noise_ref && [ -n "${_suf}" ]; then
                    # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                    _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
                fi
                # _spk_list="spk1 noise1 noise2"

                if $use_dereverb_ref && [ -n "${_suf}" ]; then
                    # references for dereverberation
                    _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
                fi
                # _spk_list="spk1 dereverb1 dereverb2"
                
                # 格式化wav.scp文件生成"spk1.scp dereverb1.scp dereverb2.scp  wav.scp "
                for spk in ${_spk_list} "wav" ; do
                    # shellcheck disable=SC2086
                    scripts/audio/format_wav_scp.sh --nj "${nj}" --cmd "${train_cmd}" \
                        --out-filename "${spk}.scp" \
                        --ref_channels "${ref_channel}" \
                        --audio-format "${audio_format}" --fs "${fs}" ${_opts} \
                        "data/${dset}/${spk}.scp" "${data_feats}${_suf}/${dset}" \
                        "${data_feats}${_suf}/${dset}/logs/${spk}" "${data_feats}${_suf}/${dset}/data/${spk}"

                done

                echo "${feats_type}" > "${data_feats}${_suf}/${dset}/feats_type"

            done

        else
            log "Error: not supported: --feats_type ${feats_type}"
            exit 2
        fi
    fi


    if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
        log "Stage 4: Remove long/short data: ${data_feats}/org -> ${data_feats}"

        # NOTE(kamo): Not applying to test_sets to keep original data
        for dset in "${train_set}" "${valid_set}"; do

            _spk_list=" "
            _scp_list=" "
            #for i in $(seq ${spk_num}); do
                #_spk_list+="spk${i} "
                #_scp_list+="spk${i}.scp "
            #done
            if $use_noise_ref; then
                # references for denoising ("noise1 noise2 ... niose${noise_type_num} ")
                _spk_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n "; done)
                _scp_list+=$(for n in $(seq $noise_type_num); do echo -n "noise$n.scp "; done)
            fi
            if $use_dereverb_ref; then
                # references for dereverberation
                _spk_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n "; done)
                _scp_list+=$(for n in $(seq $dereverb_ref_num); do echo -n "dereverb$n.scp "; done)
            fi

            # Copy data dir
            utils/copy_data_dir.sh --validate_opts --non-print "${data_feats}/org/${dset}" "${data_feats}/${dset}"
            cp "${data_feats}/org/${dset}/feats_type" "${data_feats}/${dset}/feats_type"

            for utt_extra_file in ${utt_extra_files}; do
                if [ -f "${data_feats}/org/${dset}/${utt_extra_file}" ]; then
                    cp "${data_feats}/org/${dset}/${utt_extra_file}" "${data_feats}/${dset}"
                fi
            done

            # Remove short utterances
            _feats_type="$(<${data_feats}/${dset}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _fs=$(python3 -c "import humanfriendly as h;print(h.parse_size('${fs}'))")
                _min_length=$(python3 -c "print(int(${min_wav_duration} * ${_fs}))")
                _max_length=$(python3 -c "print(int(${max_wav_duration} * ${_fs}))")

                # utt2num_samples is created by format_wav_scp.sh
                <"${data_feats}/org/${dset}/utt2num_samples" \
                    awk -v min_length="${_min_length}" -v max_length="${_max_length}" \
                        '{ if ($2 > min_length && $2 < max_length ) print $0; }' \
                        >"${data_feats}/${dset}/utt2num_samples"
                for spk in ${_spk_list} "wav"; do
                    <"${data_feats}/org/${dset}/${spk}.scp" \
                        utils/filter_scp.pl "${data_feats}/${dset}/utt2num_samples"  \
                        >"${data_feats}/${dset}/${spk}.scp"
                done
            fi

            # Remove empty text
            <"${data_feats}/org/${dset}/text" \
                awk ' { if( NF != 1 ) print $0; } ' >"${data_feats}/${dset}/text"

            # fix_data_dir.sh leaves only utts which exist in all files
            utils/fix_data_dir.sh --utt_extra_files "${utt_extra_files}" "${data_feats}/${dset}"
        done

        # shellcheck disable=SC2002
        #cat ${lm_train_text} | awk ' { if( NF != 1 ) print $0; } ' > "${data_feats}/lm_train.txt"
    fi


    if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
        if [ "${token_type}" = char ] || [ "${token_type}" = word ]; then
            log "Stage 5: Generate character level token_list from ${lm_train_text}"

            _opts="--non_linguistic_symbols ${nlsyms_txt}"

            # The first symbol in token_list must be "<blank>" and the last must be also sos/eos:
            # 0 is reserved for CTC-blank for ASR and also used as ignore-index in the other task
            ${python} -m espnet2.bin.tokenize_text  \
                --token_type "${token_type}" \
                --input "${data_feats}/lm_train.txt" --output "${token_list}" ${_opts} \
                --field 2- \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --write_vocabulary true \
                --add_symbol "${blank}:0" \
                --add_symbol "${oov}:1" \
                --add_symbol "${sos_eos}:-1"

        else
            log "Error: not supported --token_type '${token_type}'"
            exit 2
        fi

    fi
else
    log "Skip the stages for data preparation"
fi


# ========================== Data preparation is done here. ==========================

# 统计音频长度，并对语音加噪加混响
if ! "${skip_train}"; then
    if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
        _enh_asr_train_dir="${data_feats}/${train_set}"
        _enh_asr_valid_dir="${data_feats}/${valid_set}"
        log "Stage 10: ASR collect stats: train_set=${_enh_asr_train_dir}, valid_set=${_enh_asr_valid_dir}"

        _opts=
        if [ -n "${enh_asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.enh_s2t_train --print_config --optim adam
            _opts+="--config ${enh_asr_config} "
        fi

        _feats_type="$(<${_enh_asr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            _type=sound
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_enh_asr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "
        fi

        # 1. Split the key file
        _logdir="${enh_asr_stats_dir}/logdir"
        mkdir -p "${_logdir}"

        # Get the minimum number among ${nj} and the number lines of input files
        _nj=$(min "${nj}" "$(<${_enh_asr_train_dir}/${_scp} wc -l)" "$(<${_enh_asr_valid_dir}/${_scp} wc -l)")

        key_file="${_enh_asr_train_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/train.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        key_file="${_enh_asr_valid_dir}/${_scp}"
        split_scps=""
        for n in $(seq "${_nj}"); do
            split_scps+=" ${_logdir}/valid.${n}.scp"
        done
        # shellcheck disable=SC2086
        utils/split_scp.pl "${key_file}" ${split_scps}

        # 2. Generate run.sh
        log "Generate '${enh_asr_stats_dir}/run.sh'. You can resume the process from stage 10 using this script"
        mkdir -p "${enh_asr_stats_dir}"; echo "${run_args} --stage 10 \"\$@\"; exit \$?" > "${enh_asr_stats_dir}/run.sh"; chmod +x "${enh_asr_stats_dir}/run.sh"

        # 3. Submit jobs
        log "ASR collect-stats started... log: '${_logdir}/stats.*.log'"

        # NOTE: --*_shape_file doesn't require length information if --batch_type=unsorted,
        #       but it's used only for deciding the sample ids.

        # shellcheck disable=SC2086
        ${train_cmd} JOB=1:"${_nj}" "${_logdir}"/stats.JOB.log \
            ${python} -m espnet2.bin.enh_s2t_train \
                --collect_stats true \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --train_data_path_and_name_and_type "${_enh_asr_train_dir}/${_scp},speech,${_type}" \
                --train_data_path_and_name_and_type "${_enh_asr_train_dir}/text,text,text" \
                --train_data_path_and_name_and_type "${_enh_asr_train_dir}/clean.scp,speech_ref1,${_type}" \
                --valid_data_path_and_name_and_type "${_enh_asr_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_enh_asr_valid_dir}/text,text,text" \
                --input_size 80 \
                --train_shape_file "${_logdir}/train.JOB.scp" \
                --valid_shape_file "${_logdir}/valid.JOB.scp" \
                --output_dir "${_logdir}/stats.JOB" \
                ${_opts} ${enh_asr_args} || { cat "${_logdir}"/stats.1.log; exit 1; }

        # 4. Aggregate shape files
        _opts=
        for i in $(seq "${_nj}"); do
            _opts+="--input_dir ${_logdir}/stats.${i} "
        done
        # shellcheck disable=SC2086
        ${python} -m espnet2.bin.aggregate_stats_dirs ${_opts} --output_dir "${enh_asr_stats_dir}"

        # Append the num-tokens at the last dimensions. This is used for batch-bins count
        <"${enh_asr_stats_dir}/train/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${enh_asr_stats_dir}/train/text_shape.${token_type}"

        <"${enh_asr_stats_dir}/valid/text_shape" \
            awk -v N="$(<${token_list} wc -l)" '{ print $0 "," N }' \
            >"${enh_asr_stats_dir}/valid/text_shape.${token_type}"
    fi


    if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
        _enh_asr_train_dir="${data_feats}/${train_set}"
        _enh_asr_valid_dir="${data_feats}/${valid_set}"
        log "Stage 11: ASR Training: train_set=${_enh_asr_train_dir}, valid_set=${_enh_asr_valid_dir}"

        _opts=
        if [ -n "${enh_asr_config}" ]; then
            # To generate the config file: e.g.
            #   % python3 -m espnet2.bin.enh_s2t_train --print_config --optim adam
            _opts+="--config ${enh_asr_config} "
        fi

        _feats_type="$(<${_enh_asr_train_dir}/feats_type)"
        if [ "${_feats_type}" = raw ]; then
            _scp=wav.scp
            # "sound" supports "wav", "flac", etc.
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi
            _opts+="--frontend_conf fs=${fs} "
        else
            _scp=feats.scp
            _type=kaldi_ark
            _input_size="$(<${_enh_asr_train_dir}/feats_dim)"
            _opts+="--input_size=${_input_size} "

        fi
        if [ "${feats_normalize}" = global_mvn ]; then
            # Default normalization is utterance_mvn and changes to global_mvn
            _opts+="--normalize=global_mvn --normalize_conf stats_file=${enh_asr_stats_dir}/train/feats_stats.npz "
        fi
        
        # train_data_path_and_name_and_type用来创建dataset
        # num_splits_asr 大于1时
        # 语料库被分割成子集，并且每个子集被依次用于逐个训练，
        if [ "${num_splits_asr}" -gt 1 ]; then
            # If you met a memory error when parsing text files, this option may help you.
            # The corpus is split into subsets and each subset is used for training one by one in order,
            # so the memory footprint can be limited to the memory required for each dataset.

            _split_dir="${enh_asr_stats_dir}/splits${num_splits_asr}"
            # 调用划分scp文件的python文件，划分结果存放在_split_dir下。
            if [ ! -f "${_split_dir}/.done" ]; then
                rm -f "${_split_dir}/.done"
                ${python} -m espnet2.bin.split_scps \
                  --scps \
                      "${_enh_asr_train_dir}/${_scp}" \
                      "${_enh_asr_train_dir}/text" \
                      "${enh_asr_stats_dir}/train/speech_shape" \
                      "${enh_asr_stats_dir}/train/speech_ref1_shape" \
                      "${enh_asr_stats_dir}/train/text_shape.${token_type}" \
                  --num_splits "${num_splits_asr}" \
                  --output_dir "${_split_dir}"
                touch "${_split_dir}/.done"
            else
                log "${_split_dir}/.done exists. Spliting is skipped"
            fi
            # _split_dir = "exp/stat/splits2" 传入的需要是一个目录
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/spk1.scp,speech_ref1,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_split_dir}/text,text,text "
            _opts+="--train_shape_file ${_split_dir}/speech_shape "
            _opts+="--train_shape_file ${_split_dir}/speech_ref1_shape "
            _opts+="--train_shape_file ${_split_dir}/text_shape.${token_type} "
            _opts+="--multiple_iterator true "

        else
            _opts+="--train_data_path_and_name_and_type ${_enh_asr_train_dir}/${_scp},speech,${_type} "
            _opts+="--train_data_path_and_name_and_type ${_enh_asr_train_dir}/text,text,text "
            _opts+="--train_shape_file ${enh_asr_stats_dir}/train/speech_shape "
            _opts+="--train_shape_file ${enh_asr_stats_dir}/train/text_shape.${token_type} "
        fi
            #_opts+="--train_data_path_and_name_and_type ${_enh_asr_train_dir}/clean.scp,speech_ref1,${_type} "
            #_opts+="--train_shape_file ${enh_asr_stats_dir}/train/speech_ref1_shape "
        log "Generate '${enh_asr_exp}/run.sh'. You can resume the process from stage 11 using this script"
        mkdir -p "${enh_asr_exp}"; echo "${run_args} --stage 11 \"\$@\"; exit \$?" > "${enh_asr_exp}/run.sh"; chmod +x "${enh_asr_exp}/run.sh"

        # NOTE(kamo): --fold_length is used only if --batch_type=folded and it's ignored in the other case
        log "ENHANCE_ASR training started... log: '${enh_asr_exp}/train.log'"
        if echo "${cuda_cmd}" | grep -e queue.pl -e queue-freegpu.pl &> /dev/null; then
            # SGE can't include "/" in a job name
            jobname="$(basename ${enh_asr_exp})"
        else
            jobname="${enh_asr_exp}/train.log"
        fi

        # shellcheck disable=SC2086
        CUDA_VISIBLE_DEVICES=${device}\
        ${python} -m espnet2.bin.launch \
            --cmd "${cuda_cmd} --name ${jobname}" \
            --log "${enh_asr_exp}"/train.log \
            --ngpu "${ngpu}" \
            --num_nodes "${num_nodes}" \
            --init_file_prefix "${enh_asr_exp}"/.dist_init_ \
            --multiprocessing_distributed true -- \
            ${python} -m espnet2.bin.enh_s2t_train \
                --use_preprocessor true \
                --token_type "${token_type}" \
                --token_list "${token_list}" \
                --non_linguistic_symbols "${nlsyms_txt}" \
                --cleaner "${cleaner}" \
                --g2p "${g2p}" \
                --valid_data_path_and_name_and_type "${_enh_asr_valid_dir}/${_scp},speech,${_type}" \
                --valid_data_path_and_name_and_type "${_enh_asr_valid_dir}/text,text,text" \
                --valid_shape_file "${enh_asr_stats_dir}/valid/speech_shape" \
                --valid_shape_file "${enh_asr_stats_dir}/valid/text_shape.${token_type}" \
                --resume true \
                --output_dir "${enh_asr_exp}" \
                --init_param ${pretrained_model} \
                --ignore_init_mismatch ${ignore_init_mismatch} \
                --freeze_param "${freeze_param}"  \
                --input_size "${input_size}" \
                ${_opts} ${enh_asr_args}
                
                #=--valid_data_path_and_name_and_type "${_enh_asr_valid_dir}/clean.scp,speech_ref1,${_type}" \
                #=--valid_shape_file "${enh_asr_stats_dir}/valid/speech_ref1_shape" \

                #--valid_data_path_and_name_and_type "${_enh_asr_valid_dir}/clean.scp,speech_ref1,${_type}" \
                #--valid_shape_file "${enh_asr_stats_dir}/valid/speech_ref1_shape" \
                #--ignore_init_mismatch ${ignore_init_mismatch} \
    fi
else
    log "Skip the training stages"
fi
                



if ! "${skip_eval}"; then
    # ASR解码
    if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
        log "Stage 12: Decoding: training_dir=${enh_asr_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=
        if [ -n "${inference_config}" ]; then
            _opts+="--config ${inference_config} "
        fi


        # 2. Generate run.sh
        log "Generate '${enh_asr_exp}/${inference_tag}/run.sh'. You can resume the process from stage 12 using this script"
        mkdir -p "${enh_asr_exp}/${inference_tag}"; echo "${run_args} --stage 12 \"\$@\"; exit \$?" > "${enh_asr_exp}/${inference_tag}/run.sh"; chmod +x "${enh_asr_exp}/${inference_tag}/run.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${enh_asr_exp}/${inference_tag}/${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _feats_type="$(<${_data}/feats_type)"
            if [ "${_feats_type}" = raw ]; then
                _scp=wav.scp
                if [[ "${audio_format}" == *ark* ]]; then
                    _type=kaldi_ark
                else
                    _type=sound
                fi
            else
                _scp=feats.scp
                _type=kaldi_ark
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            # espnet2.bin.asr_inference 调用的是asr_inference，但是在函数里build的是asr或者enh+asr或者st模型
            asr_inference_tool="espnet2.bin.asr_inference"
            
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit decoding jobs
            log "Decoding started... log: '${_logdir}/asr_inference.*.log'"
            # shellcheck disable=SC2086
            CUDA_VISIBLE_DEVICES=${device}\
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
                ${python} -m ${asr_inference_tool} \
                    --enh_s2t_task true \
                    --batch_size ${batch_size} \
                    --ngpu "${_ngpu}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --asr_train_config "${enh_asr_exp}"/config.yaml \
                    --asr_model_file "${enh_asr_exp}"/"${inference_enh_asr_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${asr_inference_args}
                    
                    #补clean的时候加
                    #--data_path_and_name_and_type "${_data}/clean.scp,clean,${_type}" \
            # 3. Concatenates the output files from each jobs
            # 拼接多线程解码的结果，并排序
            for f in token token_int score text; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/1best_recog/${f}"
                done | LC_ALL=C sort -k1 >"${_dir}/${f}"
            done

        done
    fi

    # 保存增强语音
    if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
        log "Stage 13: Enhance Speech: training_dir=${enh_asr_exp}"

        if ${gpu_inference}; then
            _cmd="${cuda_cmd}"
            _ngpu=1
        else
            _cmd="${decode_cmd}"
            _ngpu=0
        fi

        _opts=

        # 2. Generate run.sh
        log "Generate '${enh_asr_exp}/run_enhance.sh'. You can resume the process from stage 13 using this script"
        mkdir -p "${enh_asr_exp}"; echo "${run_args} --stage 13 \"\$@\"; exit \$?" > "${enh_asr_exp}/run_enhance.sh"; chmod +x "${enh_asr_exp}/run_enhance.sh"

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${enh_asr_exp}/${inference_enh_tag}_${dset}"
            _logdir="${_dir}/logdir"
            mkdir -p "${_logdir}"

            _scp=wav.scp
            if [[ "${audio_format}" == *ark* ]]; then
                _type=kaldi_ark
            else
                _type=sound
            fi

            # 1. Split the key file
            key_file=${_data}/${_scp}
            split_scps=""
            _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
            for n in $(seq "${_nj}"); do
                split_scps+=" ${_logdir}/keys.${n}.scp"
            done
            # shellcheck disable=SC2086
            utils/split_scp.pl "${key_file}" ${split_scps}

            # 2. Submit inference jobs
            log "Enhancement started... log: '${_logdir}/enh_inference.*.log'"
            # shellcheck disable=SC2046,SC2086
            ${_cmd} --gpu "${_ngpu}" JOB=1:"${_nj}" "${_logdir}"/enh_inference.JOB.log \
                ${python} -m espnet2.bin.enh_inference \
                    --enh_s2t_task true \
                    --ngpu "${_ngpu}" \
                    --fs "${fs}" \
                    --data_path_and_name_and_type "${_data}/${_scp},speech_mix,${_type}" \
                    --key_file "${_logdir}"/keys.JOB.scp \
                    --train_config "${enh_asr_exp}"/config.yaml \
                    ${inference_enh_config:+--inference_config "$inference_enh_config"} \
                    --model_file "${enh_asr_exp}"/"${inference_enh_asr_model}" \
                    --output_dir "${_logdir}"/output.JOB \
                    ${_opts} ${enh_inference_args} || { cat $(grep -l -i error "${_logdir}"/enh_inference.*.log) ; exit 1; }

            # 3. Concatenates the output files from each jobs
            _spk_list=" "
            for i in $(seq ${spk_num}); do
                _spk_list+="spk${i} "
            done

            for spk in ${_spk_list}; do
                for i in $(seq "${_nj}"); do
                    cat "${_logdir}/output.${i}/${spk}.scp"
                done | LC_ALL=C sort -k1 > "${_dir}/${spk}.scp"
            done
        done
    fi
    
    # ASR打分
    if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
        log "Stage 14: Scoring ASR"
        if [ "${token_type}" = phn ]; then
            log "Error: Not implemented for token_type=phn"
            exit 1
        fi

        for dset in ${test_sets}; do
            _data="${data_feats}/${dset}"
            _dir="${enh_asr_exp}/${inference_tag}/${dset}"

            #for _type in cer wer ter; do
            for _type in cer; do
                [ "${_type}" = ter ] && [ ! -f "${bpemodel}" ] && continue

                _scoredir="${_dir}/score_${_type}"
                mkdir -p "${_scoredir}"

                if [ "${_type}" = wer ]; then
                    # Tokenize text to word level
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type word \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  --cleaner "${cleaner}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text"  \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type word \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"


                elif [ "${_type}" = cer ]; then
                    # Tokenize text to char level
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type char \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  --cleaner "${cleaner}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text"  \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type char \
                                  --non_linguistic_symbols "${nlsyms_txt}" \
                                  --remove_non_linguistic_symbols true \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"

                elif [ "${_type}" = ter ]; then
                    # Tokenize text using BPE
                    paste \
                        <(<"${_data}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type bpe \
                                  --bpemodel "${bpemodel}" \
                                  --cleaner "${cleaner}" \
                                ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/ref.trn"

                    # NOTE(kamo): Don't use cleaner for hyp
                    paste \
                        <(<"${_dir}/text" \
                              ${python} -m espnet2.bin.tokenize_text  \
                                  -f 2- --input - --output - \
                                  --token_type bpe \
                                  --bpemodel "${bpemodel}" \
                                  ) \
                        <(<"${_data}/utt2spk" awk '{ print "(" $2 "-" $1 ")" }') \
                            >"${_scoredir}/hyp.trn"

                fi

                sclite \
            ${score_opts} \
                    -r "${_scoredir}/ref.trn" trn \
                    -h "${_scoredir}/hyp.trn" trn \
                    -i rm -o all stdout > "${_scoredir}/result.txt"

                log "Write ${_type} result in ${_scoredir}/result.txt"
                grep -e Avg -e SPKR -m 2 "${_scoredir}/result.txt"
            done
        done

        [ -f local/score.sh ] && local/score.sh ${local_score_opts} "${enh_asr_exp}"

        # Show results in Markdown syntax
        scripts/utils/show_asr_result.sh "${enh_asr_exp}" > "${enh_asr_exp}"/RESULTS.md
        cat "${enh_asr_exp}"/RESULTS.md

    fi
    
    # 增强打分
    if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
        log "Stage 15: Scoring Enhancement"
        _cmd=${decode_cmd}

        # score_obs=true: Scoring for observation signal
        # score_obs=false: Scoring for enhanced signal
        # for score_obs in true false; do
        for score_obs in true false; do
            # Peform only at the first time for observation
            if "${score_obs}" && [ -e "${data_feats}/RESULTS_enh.md" ]; then
                log "${data_feats}/RESULTS_enh.md already exists. The scoring for observation will be skipped"
                continue
            fi

            for dset in ${test_sets}; do
                _data="${data_feats}/${dset}"
                if "${score_obs}"; then
                    _dir="${data_feats}/${dset}/scoring"
                else
                    _dir="${enh_asr_exp}/${inference_enh_tag}_${dset}/scoring"
                fi

                _logdir="${_dir}/logdir"
                mkdir -p "${_logdir}"

                # 1. Split the key file
                key_file=${_data}/wav.scp
                split_scps=""
                _nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
                for n in $(seq "${_nj}"); do
                    split_scps+=" ${_logdir}/keys.${n}.scp"
                done
                # shellcheck disable=SC2086
                utils/split_scp.pl "${key_file}" ${split_scps}

                _ref_scp=
                for spk in $(seq "${spk_num}"); do
                    _ref_scp+="--ref_scp ${_data}/spk${spk}.scp "
                done
                _inf_scp=
                for spk in $(seq "${spk_num}"); do
                    if "${score_obs}"; then
                        # To compute the score of observation, input original wav.scp
                        _inf_scp+="--inf_scp ${data_feats}/${dset}/wav.scp "
                    else
                        _inf_scp+="--inf_scp ${enh_asr_exp}/${inference_enh_tag}_${dset}/spk${spk}.scp "
                    fi
                done

                # 2. Submit scoring jobs
                log "Scoring started... log: '${_logdir}/enh_scoring.*.log'"
                # shellcheck disable=SC2086
                ${_cmd} JOB=1:"${_nj}" "${_logdir}"/enh_scoring.JOB.log \
                    ${python} -m espnet2.bin.enh_scoring \
                        --key_file "${_logdir}"/keys.JOB.scp \
                        --output_dir "${_logdir}"/output.JOB \
                        ${_ref_scp} \
                        ${_inf_scp} \
                        --ref_channel ${ref_channel}

                for spk in $(seq "${spk_num}"); do
                    for protocol in ${scoring_protocol} wav; do
                        for i in $(seq "${_nj}"); do
                            cat "${_logdir}/output.${i}/${protocol}_spk${spk}"
                        done | LC_ALL=C sort -k1 > "${_dir}/${protocol}_spk${spk}"
                    done
                done


                for protocol in ${scoring_protocol}; do
                    # shellcheck disable=SC2046
                    paste $(for j in $(seq ${spk_num}); do echo "${_dir}"/"${protocol}"_spk"${j}" ; done)  |
                    awk 'BEGIN{sum=0}
                        {n=0;score=0;for (i=2; i<=NF; i+=2){n+=1;score+=$i}; sum+=score/n}
                        END{printf ("%.2f\n",sum/NR)}' > "${_dir}/result_${protocol,,}.txt"
                done
            done

            ./scripts/utils/show_enh_score.sh "${_dir}/../.." > "${_dir}/../../RESULTS_enh.md"
        done
        log "Evaluation result for observation: ${data_feats}/RESULTS_enh.md"
        log "Evaluation result for enhancement: ${enh_asr_exp}/RESULTS_enh.md"

    fi
else
    log "Skip the evaluation stages"
fi




log "Successfully finished. [elapsed=${SECONDS}s]"
