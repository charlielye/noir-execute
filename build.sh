#!/bin/bash
set -e

IN=${1:-~/noir-projects/hello_world/target/hello_world.json}
OP=${2:-0}
LLVM=${LLVM:-16}

(cd ~/aztec-repos/barretenberg/cpp && cmake --preset clang16 -B build && cmake --build build --target libbarretenberg.a libenv.a)
(cd ~/noir-projects/hello_world && nargo compile --silence-warnings)

cargo run $IN test.out > program.ll
llc-$LLVM -O$OP -filetype=asm -relocation-model=pic -o program.s program.ll
clang++ -O$OP program.s -o program -L$HOME/aztec-repos/barretenberg/cpp/build/lib -lbarretenberg -lenv
