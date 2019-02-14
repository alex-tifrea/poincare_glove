#!/bin/bash

cython gensim/models/word2vec_inner.pyx
if [ $? -eq 0 ]; then
  python3 ./setup.py build_ext --inplace
else
  echo
  echo "ERROR: Stopped cython compilation due to previous errors."
  exit -1
fi
