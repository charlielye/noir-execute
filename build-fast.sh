#!/bin/bash
set -e

IN=${1:-hello_world}
ARCH=${ARCH:-native}
O=${O:-0}

# (cd ~/aztec-repos/barretenberg/cpp && cmake --preset clang16 -B build && cmake --build build --target libbarretenberg.a libenv.a)
(cd ~/noir-projects/$IN && nargo compile --silence-warnings)

cargo run ~/noir-projects/$IN/target/$IN.json dummy.out > program.ll

if [ "$ARCH" == "x86-64" ]; then
  ARGS="-mattr=+avx"
fi
llc-16 -O$O -march=$ARCH $ARGS -filetype=asm -relocation-model=pic -o program.s program.ll
clang++ -O$O program.s -o program -L$HOME/aztec-repos/barretenberg/cpp/build/lib -lbarretenberg -lenv
