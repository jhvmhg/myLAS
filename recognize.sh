#!/bin/bash

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;

# general configuration
backend=pytorch
stage=0        # start from 0 if you need to start from data preparation
stop_stage=100
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

train_config=conf/train.yaml
lm_config=conf/lm.yaml
decode_config=conf/decode.yaml

# rnnlm related
lm_resume=         # specify a snapshot file to resume LM training
lmtag=             # tag for managing LMs

# decoding parameter
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
n_average=10

# data dir, modify this to your AISHELL-2 data path
# tr_dir=/home/meichaoyang/workspace/git/ASR_train/data/combine_data
# dev_dir=/home/meichaoyang/dataset/data_aishell2/feats/test
test_dir=/home/meichaoyang/dataset/weixin

# exp tag
tag="" # tag for managing experiments.

. utils/parse_options.sh || exit 1;

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set=train
train_dev=dev
# recog_set="/home1/meichaoyang/dataset/magictang/espnet/test/dump"
recog_set="test"
# recog_set="espnet/test"

if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    ### Task dependent. You have to make data the following preparation part by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 0: Data preparation"
    # For training set
    mkdir -p data/train/ data/dev/ data/test/

#     cp ${tr_dir}/wav.scp ${tr_dir}/utt2spk ${tr_dir}/spk2utt ${tr_dir}/text ${tr_dir}/feats.scp data/train/ || exit 1;
#     cp ${dev_dir}/wav.scp ${dev_dir}/utt2spk ${dev_dir}/spk2utt ${dev_dir}/text ${dev_dir}/feats.scp data/dev/ || exit 1;
    cp ${test_dir}/wav.scp ${test_dir}/utt2spk ${test_dir}/spk2utt ${test_dir}/text ${test_dir}/feats.scp data/test/ || exit 1;
    # # For dev and test set
    
    # Normalize text to capital letters
    for x in test; do
        mv data/${x}/text data/${x}/text.org
        paste <(cut -f 1 data/${x}/text.org) <(cut -f 2 data/${x}/text.org | tr '[:lower:]' '[:upper:]') \
            > data/${x}/text
        rm data/${x}/text.org
    done
fi

# feat_tr_dir=${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
# feat_dt_dir=${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}
if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 1: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features; by default 80-dimensional fbanks with pitch on each frame
#     steps/make_fbank.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
#         data/train exp/make_fbank/train ${fbankdir}
#     utils/fix_data_dir.sh data/train

#         steps/make_fbank.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
#             data/dev exp/make_fbank/dev ${fbankdir}
#         utils/fix_data_dir.sh data/dev     
#         steps/make_fbank.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
#             data/test exp/make_fbank/test ${fbankdir}
#         utils/fix_data_dir.sh data/test

    
    # speed-perturbed
#     utils/perturb_data_dir_speed.sh 0.9 data/train data/temp1
#     utils/perturb_data_dir_speed.sh 1.0 data/train data/temp2
#     utils/perturb_data_dir_speed.sh 1.1 data/train data/temp3
#     utils/combine_data.sh --extra-files utt2uniq data/${train_set} data/temp1 data/temp2 data/temp3
#     rm -r data/temp1 data/temp2 data/temp3
#     steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 100 --write_utt2num_frames true \
#     data/${train_set} exp/make_fbank/${train_set} ${fbankdir}
#     utils/fix_data_dir.sh data/${train_set}

    # compute global CMVN
#     compute-cmvn-stats scp:data/${train_set}/feats.scp data/${train_set}/cmvn.ark
    
#     for rtask in ${recog_set}; do
#         steps/make_fbank.sh --cmd "$train_cmd" --nj 20 --write_utt2num_frames true \
#             data/dev exp/make_fbank/dev ${rtask}
#         utils/fix_data_dir.sh data/dev     
#     done

    # dump features for training
    split_dir=$(echo $PWD | awk -F "/" '{print $NF "/" $(NF-1)}')
    
        
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 20 --do_delta ${do_delta} \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=data/lang_1char/${train_set}_units.txt
echo "dictionary: ${dict}"

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    
        
    for rtask in ${recog_set}; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
		     data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done   
fi


# you can skip this and remove --rnnlm option in the recognition (stage 5)
if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
fi
lmexpname=train_rnnlm_${backend}_${lmtag}
lmexpdir=${PWD}/lmexpdir/${lmexpname}




if [ -z ${tag} ]; then
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
else
    expname=${train_set}_${backend}_${tag}
fi
expdir=${PWD}/${expname}
mkdir -p ${expdir}



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "stage 5: Decoding"
    nj=12
    recog_model=snapshot.ep.3
#     recog_model="model.acc.best"

    

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_avg_best_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json
        
#         echo "--------"
#         echo ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  
#             --rnnlm ${lmexpdir}/results/rnnlm.model.best

        score_sclite.sh ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
fi
