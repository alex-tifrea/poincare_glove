#!/bin/bash

alpha="0.0001 0.001 0.01 0.05 0.1"
scaling="0.001 0.01 0.1 0.5 0.75 0.9 0.999"
transform="exp id"

# for t in $transform; do
#   for lr in $alpha; do
#     if [[ $lr = 0.0001 ]]; then
#       min_alpha=0.0001
#     else
#       min_alpha=0.001
#     fi
#     ./run.sh --ds text8 --root .. --size 100 --alpha $lr --min_alpha $min_alpha \
#         --negative 5 --window 5 --poincare 1 --epochs 1 --workers 5 \
#         --restrict_vocab 0 --min_count 100 --nll \
#         --optimizer wfullrsgd --init_config ${t}0.1
#   done
# done

for s in $scaling; do
  for t in $transform; do
    echo "Scaling "$s"; Tranform "$t
    # lr = 0.05->0.001
    ./run.sh --ds text8 --root .. --size 100 --alpha 0.05 --min_alpha 0.001 \
        --negative 5 --window 5 --poincare 1 --epochs 1 --workers 10 \
        --restrict_vocab 0 --min_count 100 --nll \
        --optimizer wfullrsgd --init_config $t$s

#     # lr = 0.0001->0.0001
#     ./run.sh --ds text8 --root .. --size 100 --alpha 0.0001 --min_alpha 0.0001 \
#         --negative 5 --window 5 --poincare 1 --epochs 1 --workers 5 \
#         --restrict_vocab 0 --min_count 100 --nll \
#         --optimizer wfullrsgd --init_config $t$s
  done
done
