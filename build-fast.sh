#!/bin/bash
set -e

PROJECT_DIR=${1:-$HOME/noir-projects/hello_world}
O=${O:-0}
BB=${BB:-0}
PACKAGES=${PACKAGES:-0}
RUN=${RUN:-0}
NARGO=${NARGO:-1}
CARGO=${CARGO:-1}
ASM=${ASM:-0}
AVX=${ASM:-0}

if [ "$PACKAGES" -eq 1 ]; then
  sudo apt install -y libpolly-16-dev libz-dev libzstd-dev lldb
fi

if [ "$BB" -eq 1 ]; then
  (cd ~/aztec-repos/barretenberg/cpp && cmake --preset clang16 -B build && cmake --build build --target libbarretenberg.a libenv.a)
fi

if [ ! -d "$PROJECT_DIR/target" ] || [ "$NARGO" -eq 1 ]; then
  (cd $PROJECT_DIR && rm -rf target && nargo compile --silence-warnings --force-brillig)
fi

if [ ! -f target/release/noir-execute ] || [ "$CARGO" -eq 1 ]; then
  cargo build --release
fi

./target/release/noir-execute -p $PROJECT_DIR $TARGS > program.ll

if [ -n "$ARCH" ]; then
  ARGS+=" -march=$ARCH"
fi

if [ "$AVX" -eq 1 ]; then
  ARGS+=" -mattr=+avx"
fi

if [ "$ASM" -eq 1 ]; then
  llc-16 -O$O $ARGS -filetype=asm -relocation-model=pic -o program.s program.ll
  clang++ -O$O program.s -o program -L$HOME/aztec-repos/barretenberg/cpp/build/lib -lbarretenberg -lenv
else
  clang++ -O$O $ARGS program.ll -o program -L$HOME/aztec-repos/barretenberg/cpp/build/lib -lbarretenberg -lenv -Wno-override-module
fi

if [ "$RUN" -eq 1 ]; then
  time ./program
fi
