#!/bin/bash

export LC_CTYPE=C
random_ending=`cat /dev/urandom | tr -cd 'a-f0-9' | head -c 5`

LEDGER_FILENAME=experiments_ledger.txt

# !!! separate parameters from their values with "=" not with space, when
# calling run_glove.sh
TRAIN=true
EVAL=true
MIX=false
RESTRICT_VOCAB=400000
LR=0.05
NO_REDIRECT=false
DEBUG=false
DIST_FUNC=""
NN_CONFIG=""
COOCC_FUNC="log"
USE_SCALING=false
USE_LOG_PROBS=false
BIAS=false
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
      --root)
      ROOT="$2"
      shift # past argument
      shift # past value
      ;;
      --use_our_format)
      USE_OUR_FORMAT=true
      shift # past argument
      ;;
      --coocc_file)
      COOCC_FILE="$2"
      shift # past argument
      shift # past value
      ;;
      --vocab_file)
      VOCAB_FILE="$2"
      shift # past argument
      shift # past value
      ;;
      --train)
      TRAIN=true
      shift # past argument
      ;;
      --eval)
      TRAIN=false
      shift # past argument
      ;;
      --mix)
      MIX=true
      shift # past argument
      ;;
      --no_eval)
      TRAIN=true
      EVAL=false
      shift # past argument
      ;;
      --size)
      SIZE="$2"
      shift # past argument
      shift # past value
      ;;
      --num_embs)
      NUM_EMBS="$2"
      shift # past argument
      shift # past value
      ;;
      --lr)
      LR="$2"
      shift # past argument
      shift # past value
      ;;
      --optimizer)
      OPTIMIZER="$2"
      shift # past argument
      shift # past value
      ;;
      --coocc_func)
      COOCC_FUNC="$2"
      shift # past argument
      shift # past value
      ;;
      --dist_func)
      DIST_FUNC="$2"
      shift # past argument
      shift # past value
      ;;
      --nn_config)
      NN_CONFIG="$2"
      shift # past argument
      shift # past value
      ;;
      --use_scaling)
      USE_SCALING=true
      shift # past argument
      ;;
      --use_log_probs)
      USE_LOG_PROBS=true
      shift # past argument
      ;;
      --euclid)
      EUCLID="$2"
      shift # past argument
      shift # past value
      ;;
      --poincare)
      POINCARE="$2"
      shift # past argument
      shift # past value
      ;;
      --epochs)
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
      --workers)
      WORKERS="$2"
      shift # past argument
      shift # past value
      ;;
      --chunksize)
      CHUNKSIZE="$2"
      shift # past argument
      shift # past value
      ;;
      --restrict_vocab)
      RESTRICT_VOCAB="$2"
      shift # past argument
      shift # past value
      ;;
      --cosadd)
      COSADD=true
      shift # past argument
      ;;
      --cosmul)
      COSMUL=true
      shift # past argument
      ;;
      --distadd)
      DISTADD=true
      shift # past argument
      ;;
      --pairdir)
      PAIRDIR=true
      shift # past argument
      ;;
      --hypcosadd)
      HYPCOSADD=true
      shift # past argument
      ;;
      --agg)
      AGG=true
      shift # past argument
      ;;
      --ctx)
      CTX=true
      shift # past argument
      ;;
      --cosine_eval)
      COSINE_EVAL=true
      shift # past argument
      ;;
      --init_near_border)
      INIT_NEAR_BORDER=true
      shift # past argument
      ;;
      --init_pretrained)
      INIT_PRETRAINED=true
      shift # past argument
      ;;
      --bias)
      BIAS=true
      shift # past argument
      ;;
      --ckpt_emb)
      CKPT_EMB=true
      shift # past argument
      ;;
      --debug)
      DEBUG=true
      shift # past argument
      ;;
      --no_redirect)
      NO_REDIRECT=true
      shift # past argument
      ;;
      --model_file)
      MODEL_FILE="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

# Compile cython.
# ./make_word2vec.sh
# echo Compiled cython successfully

if [ ! -d ../train_logs ]; then
  mkdir ../train_logs
fi

if [ ! -d ../logs ]; then
  mkdir ../logs
fi

if [ ! -d ../eval_logs ]; then
  mkdir ../eval_logs
fi

if [ ! -d ../word_emb_checkpoints ]; then
  mkdir ../word_emb_checkpoints
fi

if [ ! -d ../models/glove/glove_baseline ]; then
  mkdir ../models/glove/glove_baseline
fi

if [ ! -d ../models/glove/geometric_emb ]; then
  mkdir ../models/glove/geometric_emb
fi

# Train model.
if [[ $TRAIN = true ]]; then
  # Parse command line arguments.
  MODEL_FILE="glove_ep"$EPOCHS"_size"$SIZE
  cmd="python3 glove_code/scripts/glove_main.py --train --root=.. \
      --coocc_file="$COOCC_FILE" --vocab_file="$VOCAB_FILE" --size="$SIZE"
      --workers="$WORKERS" --chunksize="$CHUNKSIZE" --epochs="$EPOCHS 
  if [[ $USE_OUR_FORMAT ]]; then
    cmd=$cmd" --use_our_format"
  fi
  if [[ $LR ]]; then
    cmd=$cmd" --lr="$LR
    MODEL_FILE=$MODEL_FILE"_lr"$LR
  fi
  if [[ $RESTRICT_VOCAB ]]; then
    cmd=$cmd" --restrict_vocab="$RESTRICT_VOCAB
    MODEL_FILE=$MODEL_FILE"_vocab"$RESTRICT_VOCAB
  fi
  if [[ $EUCLID == 1 ]]; then
    cmd=$cmd" --euclid="$EUCLID
    MODEL_FILE="../models/glove/geometric_emb/"$MODEL_FILE
    emb_type="euclid"
  elif [[ $POINCARE == 1 ]]; then
    cmd=$cmd" --poincare="$POINCARE
    MODEL_FILE="../models/glove/geometric_emb/"$MODEL_FILE
    emb_type="poincare"
  else
    MODEL_FILE="../models/glove/glove_baseline/"$MODEL_FILE
    emb_type="vanilla"
  fi
  if [[ $MIX = true ]]; then
    cmd=$cmd" --mix"
    emb_type="mix-"$emb_type
  fi
  MODEL_FILE=$MODEL_FILE"_"$emb_type
  if [[ $OPTIMIZER ]]; then
    cmd=$cmd" --optimizer "$OPTIMIZER
    MODEL_FILE=$MODEL_FILE"_OPT"$OPTIMIZER
  else
    if [[ $POINCARE == 1 ]]; then
      MODEL_FILE=$MODEL_FILE"_OPTradagrad"
    else
      MODEL_FILE=$MODEL_FILE"_OPTadagrad"
    fi
  fi
  if [[ $COOCC_FUNC != "" ]]; then
    cmd=$cmd" --coocc_func "$COOCC_FUNC
    MODEL_FILE=$MODEL_FILE"_COOCCFUNC"$COOCC_FUNC
  else
    MODEL_FILE=$MODEL_FILE"_COOCCFUNClog"
  fi
  if [[ $DIST_FUNC != "" ]]; then
    cmd=$cmd" --dist_func "$DIST_FUNC
    MODEL_FILE=$MODEL_FILE"_DISTFUNC"$DIST_FUNC
    if [[ $DIST_FUNC == "nn" ]]; then
      cmd=$cmd" --nn_config "$NN_CONFIG
      MODEL_FILE=$MODEL_FILE"_NN"$NN_CONFIG
    fi
  else
    if [[ $POINCARE == 1 ]]; then
      MODEL_FILE=$MODEL_FILE"_DISTFUNCdist-sq"
    fi
    if [[ $EUCLID == 1 ]]; then
      MODEL_FILE=$MODEL_FILE"_DISTFUNCdist"
    fi
  fi
  if [[ $NUM_EMBS ]]; then
    cmd=$cmd" --num_embs "$NUM_EMBS
    MODEL_FILE=$MODEL_FILE"_NUMEMBS"$NUM_EMBS
  fi
  if [[ $BIAS = true ]]; then
    cmd=$cmd" --bias"
    MODEL_FILE=$MODEL_FILE"_bias"
  fi
  if [[ $POINCARE == 1 && $USE_SCALING = true ]]; then
    cmd=$cmd" --use_scaling"
    MODEL_FILE=$MODEL_FILE"_scale"
  fi
  if [[ $POINCARE == 1 && $USE_LOG_PROBS = true ]]; then
    cmd=$cmd" --use_log_probs"
    MODEL_FILE=$MODEL_FILE"_logprobs"
  fi
  if [[ $INIT_NEAR_BORDER ]]; then
    cmd=$cmd" --init_near_border"
    MODEL_FILE=$MODEL_FILE"_border-init"
  fi
  if [[ $INIT_PRETRAINED ]]; then
    cmd=$cmd" --init_pretrained"
    MODEL_FILE=$MODEL_FILE"_INITpretrained"
  fi
  if [[ $CKPT_EMB = true ]]; then
    cmd=$cmd" --ckpt_emb"
  fi
  if [[ $DEBUG = true ]]; then
    cmd=$cmd" --debug"
  fi

  model_file_basename=`echo ${MODEL_FILE##*/}`
  train_log_basename="train_"`echo ${MODEL_FILE##*/}`
  train_log_file="../train_logs/"$train_log_basename

  tmp_train_log_file="../train_logs/tmp_"$random_ending
  echo "`date '+%d-%m-%Y %H:%M'` tmp_$random_ending $model_file_basename" >> $ROOT/$LEDGER_FILENAME

  echo > $tmp_train_log_file
  cmd=$cmd" --train_log_filename "$tmp_train_log_file

  echo "Training the model and preparing to save it to "$MODEL_FILE
  echo "Running "$cmd
  echo "Redirecting output to "$train_log_file

  if [[ $NO_REDIRECT != true ]]; then
    tmp_log_file="../logs/tmp_"$random_ending
    log_file=../logs/log_$model_file_basename
    eval $cmd > $tmp_log_file
    mv $tmp_log_file $log_file
  else
    eval $cmd
  fi

  if [[ $? -ne 0 ]]; then
    exit -1
  fi

  mv $tmp_train_log_file $train_log_file
fi

# Evaluate model.
if [[ $EVAL = true ]]; then
  echo "Evaluating model from "$MODEL_FILE
  cmd="python3 glove_code/scripts/glove_main.py --root=.. \
      --eval --restrict_vocab="$RESTRICT_VOCAB" \
      --model_filename="$MODEL_FILE
  if [[ $COSMUL ]]; then
    cmd=$cmd" --cosmul"
  fi
  if [[ $COSADD ]]; then
    cmd=$cmd" --cosadd"
  fi
  if [[ $DISTADD ]]; then
    cmd=$cmd" --distadd"
  fi
  if [[ $PAIRDIR ]]; then
    cmd=$cmd" --pairdir"
  fi
  if [[ $HYPCOSADD ]]; then
    cmd=$cmd" --hypcosadd"
  fi
  if [[ $AGG ]]; then
    cmd=$cmd" --agg"
  fi
  if [[ $CTX ]]; then
    cmd=$cmd" --ctx"
  fi
  if [[ $COSINE_EVAL ]]; then
    cmd=$cmd" --cosine_eval"
  fi
  if [[ $DEBUG = true ]]; then
    cmd=$cmd" --debug"
  fi
  eval $cmd
fi

