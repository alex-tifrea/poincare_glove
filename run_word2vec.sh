#!/bin/bash

export LC_CTYPE=C
random_ending=`cat /dev/urandom | tr -cd 'a-f0-9' | head -c 5`

LEDGER_FILENAME=experiments_ledger.txt

# !!! separate parameters from their values with "=" not with space, when
# calling run.sh
TRAIN=true
MIN_COUNT=100
L2REG=0.0
NO_REDIRECT=false
DEBUG=false
SIM_FUNC=""
while [[ $# -gt 0 ]]; do
  key="$1"

  case $key in
      --ds)
      DS="$2"
      shift # past argument
      shift # past value
      ;;
      --root)
      ROOT="$2"
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
      --size)
      SIZE="$2"
      shift # past argument
      shift # past value
      ;;
      --alpha)
      ALPHA="$2"
      shift # past argument
      shift # past value
      ;;
      --min_alpha)
      MIN_ALPHA="$2"
      shift # past argument
      shift # past value
      ;;
      --negative)
      NEGATIVE="$2"
      shift # past argument
      shift # past value
      ;;
      --window)
      WINDOW="$2"
      shift # past argument
      shift # past value
      ;;
      --min_count)
      MIN_COUNT="$2"
      shift # past argument
      shift # past value
      ;;
      --l2reg)
      L2REG="$2"
      shift # past argument
      shift # past value
      ;;
      --optimizer)
      OPTIMIZER="$2"
      shift # past argument
      shift # past value
      ;;
      --sim_func)
      SIM_FUNC="$2"
      shift # past argument
      shift # past value
      ;;
      --init_config)
      INIT_CONFIG="$2"
      shift # past argument
      shift # past value
      ;;
      --normalized)
      NORMALIZED=true
      shift # past argument
      ;;
      --nll)
      IS_NLL=true
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
      --torus)
      TORUS="$2"
      shift # past argument
      shift # past value
      ;;
      --epochs)
      EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
      --burnin_epochs)
      BURNIN_EPOCHS="$2"
      shift # past argument
      shift # past value
      ;;
      --workers)
      WORKERS="$2"
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
      --agg)
      AGG=true
      shift # past argument
      ;;
      --ctx)
      CTX=true
      shift # past argument
      ;;
      --shift_origin)
      SHIFT_ORIGIN=true
      shift # past argument
      ;;
      --cosine_eval)
      COSINE_EVAL=true
      shift # past argument
      ;;
      --bias)
      BIAS=true
      shift # past argument
      ;;
      --init_near_border)
      INIT_NEAR_BORDER=true
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

# Parse command line arguments.
if [[ $MODEL_FILE ]]; then
  model_file=$MODEL_FILE

  # Continue training model.
  if [[ $TRAIN = true ]]; then
    basename=${model_file##*/}
    old_epochs=`echo $basename | cut -d_ -f4`
    new_epochs=$(($EPOCHS + $old_epochs))
    new_model_file=`echo $basename | awk 'BEGIN{FS=OFS="_"} {$4 = "'$new_epochs'"; print}'`
    new_model_file=`echo $new_model_file | awk 'BEGIN{FS=OFS="_"} {$6 = "'A${ALPHA:2}'"; print}'`
    new_model_file=`echo $new_model_file | awk 'BEGIN{FS=OFS="_"} {$7 = "'a${MIN_ALPHA:2}'"; print}'`
    new_model_file=$new_model_file"_cont"

    train_log_basename=`echo ${model_file##*/} | sed 's/w2v_/train_/'`
    train_log_file="../train_logs/"$train_log_basename
    echo "tmp_$random_ending `date '+%Y-%m-%d %H:%M:%S'` $basename (continue training)" >> $ROOT/$LEDGER_FILENAME

    new_train_log_basename=`echo ${new_model_file##*/} | sed 's/w2v_/train_/'`
    new_train_log_file="../train_logs/"$new_train_log_basename

    echo "From previous training ($basename):" > $new_train_log_file
    cat $train_log_file >> $new_train_log_file

    model_file=${model_file%/*}/$new_model_file

    cmd="python3 gensim/scripts/word2vec_main.py --train --root=.. \
        --workers="$WORKERS" --epochs="$EPOCHS" \
        --alpha="$ALPHA" --min_alpha="$MIN_ALPHA" \
        --optimizer="$OPTIMIZER" \
        --sim_func="$SIM_FUNC" \
        --init_config="$INIT_CONFIG" \
        --model_filename "$MODEL_FILE" \
        --train_log_filename "$new_train_log_file
    echo "Continue training model "$MODEL_FILE
    echo "Running "$cmd
    echo "Redirecting output to "$new_train_log_file

    printf "\n==============================================================\n\n" >> $new_train_log_file 

    if [[ $NO_REDIRECT != true ]]; then
      tmp_log_file=../logs/tmp_$random_ending
      log_file=../logs/log_$new_model_file
      eval $cmd > $tmp_log_file
      mv $tmp_log_file $log_file
    else
      eval $cmd
    fi

    if [[ $? -ne 0 ]]; then
      exit -1
    fi
  fi
else
  if [[ $IS_NLL ]]; then
    model_file="w2v_"$DS"_nll_"$EPOCHS"_"$SIZE
  else
    model_file="w2v_"$DS"_sg_"$EPOCHS"_"$SIZE
  fi
  cmd="python3 gensim/scripts/word2vec_main.py --train --root=.. \
      --sg=1 --ds="$DS" --size="$SIZE" --workers="$WORKERS" --epochs="$EPOCHS 
  if [[ $ALPHA ]]; then
    cmd=$cmd" --alpha="$ALPHA
    dec="$(echo "$ALPHA" | cut -d'.' -f 2)"
    model_file=$model_file"_A"$dec
  fi
  if [[ $MIN_ALPHA ]]; then
    cmd=$cmd" --min_alpha="$MIN_ALPHA
    dec="$(echo "$MIN_ALPHA" | cut -d'.' -f 2)"
    model_file=$model_file"_a"$dec
  fi
  if [[ $NEGATIVE ]]; then
    cmd=$cmd" --negative="$NEGATIVE
    model_file=$model_file"_n"$NEGATIVE
  fi
  if [[ $WINDOW ]]; then
    cmd=$cmd" --window="$WINDOW
    model_file=$model_file"_w"$WINDOW
  fi
  if [[ $MIN_COUNT ]]; then
    cmd=$cmd" --min_count="$MIN_COUNT
    model_file=$model_file"_c"$MIN_COUNT
  fi
  if [[ $EUCLID == 1 ]]; then
    cmd=$cmd" --euclid="$EUCLID
    model_file="../models/geometric_emb/"$model_file"_euclid"
  elif [[ $POINCARE == 1 ]]; then
    cmd=$cmd" --poincare="$POINCARE
    model_file="../models/geometric_emb/"$model_file"_poincare"
  elif [[ $TORUS == 1 ]]; then
    cmd=$cmd" --torus="$TORUS
    model_file="../models/geometric_emb/"$model_file"_torus"
  else
    model_file="../models/word2vec_baseline/"$model_file"_cosine"
  fi
  if [[ $OPTIMIZER ]]; then
    cmd=$cmd" --optimizer "$OPTIMIZER
    model_file=$model_file"_OPT"$OPTIMIZER
  else
    if [[ $POINCARE == 1 ]]; then
      model_file=$model_file"_OPTrsgd"
    else
      model_file=$model_file"_OPTsgd"
    fi
  fi
  if [[ $SIM_FUNC != "" ]]; then
    cmd=$cmd" --sim_func "$SIM_FUNC
    model_file=$model_file"_SIM"$SIM_FUNC
  else
    if [[ $POINCARE == 1 ]]; then
      model_file=$model_file"_SIMdist-sq"
    fi
  fi
  if [[ $INIT_CONFIG ]]; then
    cmd=$cmd" --init_config "$INIT_CONFIG
    model_file=$model_file"_INIT"$INIT_CONFIG
  fi
  if [[ $L2REG != 0.0 ]]; then
    cmd=$cmd" --l2reg "$L2REG
    model_file=$model_file"_l"$L2REG
  fi
  if [[ $BIAS ]]; then
    cmd=$cmd" --bias"
    model_file=$model_file"_bias"
  fi
  if [[ $INIT_NEAR_BORDER ]]; then
    cmd=$cmd" --init_near_border"
    model_file=$model_file"_border-init"
  fi
  if [[ $NORMALIZED = true ]]; then
    cmd=$cmd" --normalized"
    model_file=$model_file"_norm"
  fi
  if [[ $BURNIN_EPOCHS -gt 0 ]]; then
    cmd=$cmd" --burnin_epochs "$BURNIN_EPOCHS
    model_file=$model_file"_burnin"$BURNIN_EPOCHS
  fi
  if [[ $IS_NLL ]]; then
    cmd=$cmd" --nll"
  fi
  if [[ $CKPT_EMB = true ]]; then
    cmd=$cmd" --ckpt_emb"
  fi
  if [[ $DEBUG = true ]]; then
    cmd=$cmd" --debug"
  fi

  model_file_basename=`echo ${model_file##*/}`
  train_log_basename=`echo ${model_file##*/} | sed 's/w2v_/train_/'`
  train_log_file="../train_logs/"$train_log_basename

  tmp_train_log_file="../train_logs/tmp_"$random_ending
  echo "`date '+%d-%m-%Y %H:%M'` tmp_$random_ending $model_file_basename" >> $ROOT/$LEDGER_FILENAME

  echo > $tmp_train_log_file
  cmd=$cmd" --train_log_filename "$tmp_train_log_file

  # Train model.
  if [[ $TRAIN = true ]]; then
    echo "Training the model and preparing to save it to "$model_file
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
fi

# Evaluate model.
echo "Evaluating model from "$model_file
cmd="python3 gensim/scripts/word2vec_main.py --root=.. \
    --workers="$WORKERS" --eval --restrict_vocab="$RESTRICT_VOCAB" \
    --model_filename="$model_file
if [[ $COSMUL ]]; then
  cmd=$cmd" --cosmul"
fi
if [[ $COSADD ]]; then
  cmd=$cmd" --cosadd"
fi
if [[ $AGG ]]; then
  cmd=$cmd" --agg"
fi
if [[ $CTX ]]; then
  cmd=$cmd" --ctx"
fi
if [[ $SHIFT_ORIGIN ]]; then
  cmd=$cmd" --shift_origin"
fi
if [[ $COSINE_EVAL ]]; then
  cmd=$cmd" --cosine_eval"
fi
if [[ $DEBUG = true ]]; then
  cmd=$cmd" --debug"
fi
eval $cmd
