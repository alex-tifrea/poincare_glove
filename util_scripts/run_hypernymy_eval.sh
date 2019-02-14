#!/bin/bash

for filename in "$@"; do

  echo ${filename##*/}

#   echo "WordNet 20+20"
#   python3 util_scripts/lexical_entailment_eval.py --root .. \
#     --restrict_vocab 200000 --model_filename ${filename} \
#     --words_to_use 20
#   echo
# 
#   echo "WordNet 400+400"
#   python3 util_scripts/lexical_entailment_eval.py --root .. \
#     --restrict_vocab 200000 --model_filename ${filename} \
#     --words_to_use 400
#   echo

  echo "Unsupervised 1k+1k"
  python3 util_scripts/lexical_entailment_eval.py --root .. \
    --restrict_vocab 200000 --model_filename ${filename} \
    --words_to_use 1000 --unsupervised
  echo

  echo "Unsupervised 2.5k+2.5k"
  python3 util_scripts/lexical_entailment_eval.py --root .. \
    --restrict_vocab 200000 --model_filename ${filename} \
    --words_to_use 2500 --unsupervised

  echo "Unsupervised 1k+1k"
  python3 util_scripts/lexical_entailment_eval.py --root .. \
    --restrict_vocab 200000 --model_filename ${filename} \
    --words_to_use 1000 --unsupervised
  echo

  echo "Unsupervised 2.5k+2.5k"
  python3 util_scripts/lexical_entailment_eval.py --root .. \
    --restrict_vocab 200000 --model_filename ${filename} \
    --words_to_use 2500 --unsupervised
  echo

  echo "Unsupervised 5k+5k"
  python3 util_scripts/lexical_entailment_eval.py --root .. \
    --restrict_vocab 200000 --model_filename ${filename} \
    --words_to_use 5000 --unsupervised
  echo

  echo "Unsupervised 10k+10k"
  python3 util_scripts/lexical_entailment_eval.py --root .. \
    --restrict_vocab 200000 --model_filename ${filename} \
    --words_to_use 10000 --unsupervised
  echo -e "==================================================================\n"
done

