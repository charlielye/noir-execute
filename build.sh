#!/bin/bash
set -e

PROJECT_DIR=${1:-$HOME/noir-projects/hello_world}
O=${O:-0}
BB=${BB:-0}
PACKAGES=${PACKAGES:-0}
RUN=${RUN:-0}
NARGO=${NARGO:-1}
CARGO=${CARGO:-1}
CARGO_DEBUG=${CARGO_DEBUG:-0}
ASM=${ASM:-0}
AVX=${ASM:-0}
BB_DEBUG=${BB_DEBUG:-0}
VERBOSE=${VERBOSE:-0}
TRAP=${TRAP:-0}

bb_dir=build
bb_preset=clang16
if [ "$BB_DEBUG" -eq 1 ]; then
  bb_dir=build-dbg
  bb_preset=clang16-dbg
fi

# Ensures required packages are installed. Useful for developer sysboxes that sometimes reboot.
if [ "$PACKAGES" -eq 1 ]; then
  sudo apt install -y libpolly-16-dev libz-dev libzstd-dev lldb
fi

# Build required barretenberg libs first. Useful if working on blackboxes.
if [ "$BB" -eq 1 ]; then
  (cd ~/aztec-repos/barretenberg/cpp && cmake --preset $bb_preset -B $bb_dir && cmake --build $bb_dir --target libecc.a libenv.a libcrypto.a libcommon.a libnumeric.a)
fi

# If the noir project hasn't been compiled (or forced) compile it pure brillig.
if [ ! -d "$PROJECT_DIR/target" ] || [ "$NARGO" -eq 1 ]; then
  (cd $PROJECT_DIR && rm -rf target && ~/aztec-repos/aztec-packages/noir/noir-repo/target/release/nargo compile --silence-warnings --force-brillig)
fi

[ "$ASM" -eq 1 ] && TARGS+=" --write-ll"      # Write the .ll file instead of .o file after transpile.
[ "$TRAP" -eq 1 ] && TARGS+=" --ill-trap"     # Cause program to crash rather than exit on traps. Useful for debugging.
[ "$VERBOSE" -eq 1 ] && TARGS+=" --verbose"   # Print all the program opcodes and their locations, basic block numbers, etc.
[ "$AVX" -eq 1 ] && TARGS+=" --avx"           # Enable AVX in output bytecode. Presumably only useful for x86.

# Build the transpiler either as debug or release, and transpile the program.
if [ "$CARGO_DEBUG" -eq 1 ]; then
  if [ ! -f target/debug/noir-execute ] || [ "$CARGO" -eq 1 ]; then
    cargo build
  fi
  ./target/debug/noir-execute -p $PROJECT_DIR $TARGS
else
  if [ ! -f target/release/noir-execute ] || [ "$CARGO" -eq 1 ]; then
    cargo build --release
  fi
  ./target/release/noir-execute -p $PROJECT_DIR $TARGS
fi

[ -n "$ARCH" ] && ARGS+=" -march=$ARCH"
[ "$AVX" -eq 1 ] && ARGS+=" -mattr=+avx"

# Compile the program, either from .ll to .s then linking, or from the .o file.
# We link with the required barretenberg libs.
libs="-lecc -lenv -lcrypto -lcommon -lnumeric"
if [ "$ASM" -eq 1 ]; then
  llc-16 -O$O $ARGS -filetype=asm -relocation-model=pic -o program.s program.ll
  clang++ -O$O program.s -o program -L$HOME/aztec-repos/barretenberg/cpp/$bb_dir/lib $libs
else
  clang++ -O$O $ARGS program.o -o program -L$HOME/aztec-repos/barretenberg/cpp/$bb_dir/lib -Wno-override-module $libs
fi

# If requested, run the program and time it.
if [ "$RUN" -eq 1 ]; then
  echo "Running..."
  time ./program
fi
