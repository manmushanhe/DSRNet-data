#!/bin/bash
#
# Copyright 2017  Brno University of Technology (Author: Karel Vesely);
# Apache 2.0

# This scripts splits 'data' directory into two parts:
# - training set with 90% of speakers
# - held-out set with 10% of speakers (cv)
# (to be used in frame cross-entropy training of 'nnet1' models),

# The script also accepts a list of held-out set speakers by '--cv-spk-list'
# (with perturbed data, we pass the list of speakers externally).
# The remaining set of speakers is the the training set.

cv_spk_percent=10
cv_spk_list= # To be used with perturbed data,
seed=777
cv_utt_percent=10 # ignored (compatibility),
. utils/parse_options.sh

if [ $# != 3 ]; then
  echo "Usage: $0 [opts] <src-data> <train-data> <cv-data>"
  echo "  --cv-spk-percent N (default 10)"
  echo "  --cv-spk-list <file> (a pre-defined list with cv speakers)"
  exit 1;
fi

set -euo pipefail

src_data=$1
trn_data=$2
cv_data=$3

#[ ! -r $src_data/spk2utt ] && echo "Missing '$src_data/spk2utt'. Error!" && exit 1

tmp=/Work21/2021/luhaoyu/enviroment_new/espnet/egs2/aishell/ssl1

#if [ -z "$cv_spk_list" ]; then
  # Select 'cv_spk_percent' speakers randomly,
  #cat $src_data/spk2utt | awk '{ print $1; }' | utils/shuffle_list.pl --srand $seed >$tmp/speakers
  #n_spk=$(wc -l <$tmp/speakers)
  #n_spk_cv=$(perl -e "print int($cv_spk_percent * $n_spk / 100); ")
  #
  #head -n $n_spk_cv $tmp/speakers >$tmp/speakers_cv
  #tail -n+$((n_spk_cv+1)) $tmp/speakers >$tmp/speakers_trn
#else
  # Use pre-defined list of speakers,
  #cp $cv_spk_list $tmp/speakers_cv
  #join -v2 <(sort $cv_spk_list) <(awk '{ print $1; }' <$src_data/spk2utt | sort) >$tmp/speakers_trn
#fi

#if [ -z "$cv_spk_list" ]; then
  # Select 'cv_spk_percent' speakers randomly,
  #cat $src_data/segments | awk '{ print $1; }' | utils/shuffle_list.pl --srand $seed >$tmp/segments
  #n_utt=$(wc -l <$tmp/segments)
  #n_utt_cv=$(perl -e "print int($cv_utt_percent * $n_utt / 100); ")
  #
  #head -n $n_utt_cv $tmp/segments >$tmp/segments_cv
  #tail -n+$((n_utt_cv+1)) $tmp/segments >$tmp/segments_trn
#else
  # Use pre-defined list of speakers,
  #cp $cv_spk_list $tmp/speakers_cv
  #join -v2 <(sort $cv_spk_list) <(awk '{ print $1; }' <$src_data/spk2utt | sort) >$tmp/speakers_trn
#fi

if [ -z "$cv_spk_list" ]; then
  # Select 'cv_spk_percent' speakers randomly,
  cat $src_data/wav.scp | awk '{ print $0; }' | utils/shuffle_list.pl --srand $seed >$tmp/wav.scp
  n_utt=$(wc -l <$tmp/wav.scp)
  n_utt_cv=$(perl -e "print int($cv_utt_percent * $n_utt / 100); ")
  #
  head -n $n_utt_cv $tmp/wav.scp >$tmp/wav_cv.scp
  tail -n+$((n_utt_cv+1)) $tmp/wav.scp >$tmp/wav_trn.scp
else
  # Use pre-defined list of speakers,
  cp $cv_spk_list $tmp/wav_cv.scp
  join -v2 <(sort $cv_spk_list) <(awk '{ print $1; }' <$src_data/spk2utt | sort) >$tmp/speakers_trn
fi


# Sanity checks,
n_utt=$(wc -l <$src_data/wav.scp)
echo "wav.scp, src=$n_utt, trn=$(wc -l <$tmp/wav_trn.scp), cv=$(wc -l $tmp/wav_cv.scp)"
overlap=$(join <(sort $tmp/wav_trn.scp) <(sort $tmp/wav_cv.scp) | wc -l)
[ $overlap != 0 ] && \
  echo "WARNING, speaker overlap detected!" && \
  join <(sort $tmp/wav_trn.scp) <(sort $tmp/wav_cv.scp) | head && \
  echo '...'

# Create new data dirs,
#utils/data/subset_data_dir.sh --utt-list $tmp/wav_trn.scp $src_data $trn_data
#utils/data/subset_data_dir.sh --utt-list $tmp/wav_cv.scp $src_data $cv_data

