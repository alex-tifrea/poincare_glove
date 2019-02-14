#!/bin/bash

unamestr=`uname`

if [[ $unamestr == "Darwin" ]]; then
  export CC=/usr/local/bin/gcc-7
fi

cd glove_code
cython src/glove_inner.pyx
if [ $? -eq 0 ]; then
  python3 ./setup.py build_ext --inplace
else
  echo
  echo "ERROR: Stopped cython compilation due to previous errors."
  exit -1
fi
cd ..
