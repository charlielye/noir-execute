#!/bin/bash
set -o pipefail

#PROJECTS=${1:-~/aztec-repos/noir/test_programs/execution_success/*}

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

for DIR in $@; do
  echo -n "$(basename $DIR): "
  AVX=1 NARGO=0 CARGO=0 RUN=0 ./build-fast.sh $DIR > /dev/null 2>&1

  if [ $? -ne 0 ]; then
    echo -e "${YELLOW}FAILED TRANSPILE${NC}"
    continue
  fi

  ./program > /dev/null 2>&1

  if [ $? -ne 0 ]; then
    echo -e "${RED}FAILED${NC}"
    continue
  fi

  output=$(perf stat -e duration_time -r 10 ./program 2>&1)
  micros=$(echo "$output" | awk '/ ns/ {print int($1/1000)}')
  echo -e "${GREEN}PASSED${NC} (${micros}us)"
done
