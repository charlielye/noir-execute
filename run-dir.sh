#!/bin/bash
set -o pipefail

RED='\033[0;31m'
YELLOW='\033[0;33m'
GREEN='\033[0;32m'
NC='\033[0m'

failed_transpile=0
failed_execution=0
success=0

for DIR in $@; do
  echo -n "$(basename $DIR): "
  AVX=1 NARGO=0 CARGO=0 RUN=0 ./build.sh $DIR > /dev/null 2>&1

  if [ $? -ne 0 ]; then
    echo -e "${YELLOW}FAILED TRANSPILE${NC}"
    ((failed_transpile++))
    continue
  fi

  output=$(./program > /dev/null 2>&1)

  if [ $? -ne 0 ] || echo "$output" | grep -qi "segmentation fault"; then
    echo -e "${RED}FAILED${NC}"
    ((failed_execution++))
    continue
  fi

  output=$(perf stat -e duration_time -r 3 ./program 2>&1)
  micros=$(echo "$output" | awk '/ ns/ {print int($1/1000)}')
  echo -e "${GREEN}PASSED${NC} (${micros}us)"
  ((success++))
done

echo
echo Summary:
echo -e "         Success: ${GREEN}$success${NC}"
echo -e "Failed transpile: ${YELLOW}$failed_transpile${NC}"
echo -e "Failed execution: ${RED}$failed_execution${NC}"