#!/bin/bash
set -e

PROJECT_DIR=${1:-$HOME/noir-projects/hello_world}
O=${O:-0}
PACKAGES=${PACKAGES:-0}
RUN=${RUN:-0}
NARGO=${NARGO:-1}
SSA=${SSA:-0}
CARGO=${CARGO:-1}
CARGO_DEBUG=${CARGO_DEBUG:-0}
TRANSPILE=${TRANSPILE:-1}
ASM=${ASM:-0}
LL=${LL:-1}
AVX=${AVX:-0}
BB_DEBUG=${BB_DEBUG:-0}
VERBOSE=${VERBOSE:-0}
TRAP=${TRAP:-0}
EXE=${EXE:-1}
BACKEND=${BACKEND:-bb}
BUILD_BACKEND=${BUILD_BACKEND:-0}
PERF=${PERF:-0}

bb_dir=build
bb_preset=clang16
if [ "$BB_DEBUG" -eq 1 ]; then
  bb_dir=build-dbg
  bb_preset=clang16-dbg
fi

# Ensures required packages are installed. Useful for developer sysboxes that sometimes reboot.
if [ "$PACKAGES" -eq 1 ]; then
  sudo apt install -y libpolly-16-dev libz-dev libzstd-dev lldb
  exit
fi

# Build required libs first. Useful if working on blackboxes.
if [ "$BACKEND" == "bb" ]; then
  libs="-lecc -lenv -lcrypto -lcommon -lnumeric"
  lib_path=$HOME/aztec-repos/barretenberg/cpp/$bb_dir/lib
  if [ "$BUILD_BACKEND" -eq 1 ]; then
    (cd ~/aztec-repos/barretenberg/cpp && cmake --preset $bb_preset -B $bb_dir -DDISABLE_ASM=ON && cmake --build $bb_dir --target libecc.a libenv.a libcrypto.a libcommon.a libnumeric.a) || exit 1
  fi
elif [ "$BACKEND" == "zb" ]; then
  libs="-lziegenberg"
  lib_path=$HOME/ziegenberg/zig-out/lib
  if [ "$BUILD_BACKEND" -eq 1 ]; then
    # (cd ~/ziegenberg && zig build --release=fast) || exit 1
    (cd ~/ziegenberg && zig build) || exit 1
  fi
else
  echo "Unknown backend: $BACKEND"
  exit 1
fi

# If the noir project hasn't been compiled (or forced) compile it pure brillig.
if [ ! -d "$PROJECT_DIR/target" ] || [ "$NARGO" -eq 1 ]; then
  [ "$SSA" -eq 1 ] && NARGO_ARGS="--show-ssa"
  # (cd ~/aztec-repos/aztec-packages/noir/noir-repo && cargo build --release --bin nargo) || exit 1
  (cd $PROJECT_DIR && rm -rf target && ~/aztec-repos/aztec-packages/noir/noir-repo/target/release/nargo compile $NARGO_ARGS --silence-warnings --force-brillig) || exit 1
fi

[ "$ASM" -eq 1 ] && TARGS+=" --write-ll"      # Write the .ll file instead of .o file after transpile.
[ "$TRAP" -eq 1 ] && TARGS+=" --ill-trap"     # Cause program to crash rather than exit on traps. Useful for debugging.
[ "$VERBOSE" -eq 1 ] && TARGS+=" --verbose"   # Print all the program opcodes and their locations, basic block numbers, etc.
[ "$AVX" -eq 1 ] && TARGS+=" --avx"           # Enable AVX in output bytecode. Presumably only useful for x86.

# Build the transpiler either as debug or release, and transpile the program.
if [ "$TRANSPILE" -eq 1 ]; then
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
fi

[ -n "$ARCH" ] && ARGS+=" -march=$ARCH"
# SSE/AVX hurts us.
if [ "$AVX" -eq 1 ]; then
  ARGS+=" -mattr=+avx"
else
  ARGS+=" -mattr=-avx,-avx2,-sse"
fi

# Compile the program, either from .ll to .s then linking, or from the .o file.
if [ "$ASM" -eq 1 ]; then
  [ "$LL" -eq 1 ] && llc-16 -O$O $ARGS -filetype=asm -relocation-model=pic -o program.s program.ll
  [ "$EXE" -eq 1 ] && clang -O$O -c program.s -o program.o
  [ "$EXE" -eq 1 ] && clang++ -O$O program.o -o program -L$lib_path $libs
else
  clang++ -O$O program.o -o program -L$lib_path -Wno-override-module $libs
fi

# If requested, run the program and time it.
if [ "$RUN" -eq 1 ]; then
  echo "Running..."
  if [ "$PERF" -eq 1 ]; then
    perf record -g ./program && perf report -g
  else
    time ./program
  fi
fi
