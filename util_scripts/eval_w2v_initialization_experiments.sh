#!/bin/bash

scaling="0.001 0.01 0.1 0.5 0.75 0.9 0.999"
transform="exp id"
alpha="0001 001 01 05 1"

for t in $transform; do
  for lr in $alpha; do
    if [[ $lr = 0001 ]]; then
      min_alpha=0001
    else
      min_alpha=001
    fi
    ./run.sh --eval --workers 1 --restrict_vocab 500000 --model_file \
      ../models/geometric_emb/w2v_text8_nll_1_100_A${lr}_a${min_alpha}_n5_w5_c100_poincare_OPTwfullrsgd_INIT${t}0.1 \
      --cosine_eval
  done
done

for s in $scaling; do
  for t in $transform; do
    echo "Scaling "$s"; Tranform "$t
    # lr = 0.05->0.001
    ./run.sh --eval --workers 1 --restrict_vocab 500000 --model_file \
      ../models/geometric_emb/w2v_text8_nll_1_100_A05_a001_n5_w5_c100_poincare_OPTwfullrsgd_INIT${t}${s} \
      --cosine_eval

    # lr = 0.0001->0.0001
    ./run.sh --eval --workers 1 --restrict_vocab 500000 --model_file \
      ../models/geometric_emb/w2v_text8_nll_1_100_A0001_a0001_n5_w5_c100_poincare_OPTwfullrsgd_INIT${t}${s} \
      --cosine_eval
  done
done
