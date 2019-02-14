#!/bin/bash

HOURS=4
CORES=15
MEM=2048

model_path_pattern=${1}

echo "Evaluating models from ${model_path_pattern}"

for file in ${model_path_pattern}; do
  basename=${file##*/}
  echo "Starting jobs for evaluating ${basename}"
  bsub -W ${HOURS}:00 -n ${CORES} -R "rusage[mem=${MEM}]" -oo ../outputs/eval/out_${basename} \
    "./run_glove.sh --eval --root .. --restrict_vocab 200000 --model_file ${file}"
  bsub -W ${HOURS}:00 -n ${CORES} -R "rusage[mem=${MEM}]" -oo ../outputs/eval/out_${basename}_agg \
    "./run_glove.sh --eval --root .. --restrict_vocab 200000 --model_file ${file} --agg"
  bsub -W ${HOURS}:00 -n ${CORES} -R "rusage[mem=${MEM}]" -oo ../outputs/eval/out_${basename}_cos \
    "./run_glove.sh --eval --root .. --restrict_vocab 200000 --model_file ${file} --cosine_eval"
  bsub -W ${HOURS}:00 -n ${CORES} -R "rusage[mem=${MEM}]" -oo ../outputs/eval/out_${basename}_cos_agg \
    "./run_glove.sh --eval --root .. --restrict_vocab 200000 --model_file ${file} --cosine_eval --agg"
done
