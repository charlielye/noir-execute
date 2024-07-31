#!/bin/bash
set -e

PROJECT_DIR=${1:-$HOME/noir-projects/hello_world}
O=${O:-0}
BB=${BB:-0}
PACKAGES=${PACKAGES:-0}
RUN=${RUN:-0}

if [ "$PACKAGES" -eq 1 ]; then
  sudo apt install -y libpolly-16-dev libz-dev libzstd-dev
fi

if [ "$BB" -eq 1 ]; then
  (cd ~/aztec-repos/barretenberg/cpp && cmake --preset clang16 -B build && cmake --build build --target libbarretenberg.a libenv.a)
fi

(cd $PROJECT_DIR && nargo compile --silence-warnings)

cargo run -- -p $PROJECT_DIR > program.ll

if [ -n "$ARCH" ]; then
  ARGS+="-march=$ARCH"
  if [ "$ARCH" == "x86-64" ]; then
    ARGS+="-mattr=+avx"
  fi
fi
llc-16 -O$O $ARGS -filetype=asm -relocation-model=pic -o program.s program.ll
clang++ -O$O program.s -o program -L$HOME/aztec-repos/barretenberg/cpp/build/lib -lbarretenberg -lenv

if [ "$RUN" -eq 1 ]; then
  time ./program
fi
