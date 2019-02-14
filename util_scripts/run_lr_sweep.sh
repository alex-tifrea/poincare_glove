#!/bin/bash

LEDGER=~/queue_ledger.txt
echo -n "" > $LEDGER

lr_2D_list="0.001 0.005 0.01 0.05 0.1"
lr_100D_list="0.005 0.01 0.05 0.1"

sim_list="log-dist log-dist-sq"

# Run for 2D
for lr in $lr_2D_list; do
  echo "[2D] Launching for sim exp-dist and lr "$lr"..." >> $LEDGER
  bsub -W 24:00 -n 10 -oo ../outputs/poincare_2D_3ep_SIMexp-dist_LR${lr} "./run.sh \
    --ds levy --root .. --size 2 --alpha $lr --min_alpha 0.0001 --negative 5 \
    --window 5 --poincare 1 --epochs 3 --workers 10 --restrict_vocab 0 \
    --min_count 100 --nll --optimizer wfullrsgd --burnin_epochs 1 \
    --sim_func exp-dist" | tail -n 1 >> $LEDGER
done

# Run for 100D and exp-dist
for lr in $lr_100D_list; do
  echo "[100D] Launching for sim exp-dist and lr "$lr"..." >> $LEDGER
  bsub -W 24:00 -n 10 -oo ../outputs/poincare_100D_SIMexp-dist_LR${lr} "./run.sh \
    --ds levy --root .. --size 100 --alpha $lr --min_alpha 0.0001 --negative 5 \
    --window 5 --poincare 1 --epochs 5 --workers 10 --restrict_vocab 0 \
    --min_count 100 --nll --optimizer wfullrsgd --burnin_epochs 1 \
    --sim_func exp-dist" | tail -n 1 >> $LEDGER
done

# # Run for 100D and log-dist-sq
# for lr in $lr_100D_list; do
#   echo "[100D] Launching for sim log-dist-sq and lr "$lr"..." >> $LEDGER
#   bsub -W 24:00 -n 10 -oo ../outputs/poincare_100D_SIMlog-dist-sq_LR${lr} "./run.sh \
#     --ds levy --root .. --size 100 --alpha $lr --min_alpha 0.0001 --negative 5 \
#     --window 5 --poincare 1 --epochs 5 --workers 10 --restrict_vocab 0 \
#     --min_count 100 --nll --optimizer wfullrsgd --burnin_epochs 1 \
#     --sim_func log-dist-sq" | tail -n 1 >> $LEDGER
# done
