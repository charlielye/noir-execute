#!/bin/bash

shopt -s extglob

# 4709 produces so much bytecode it's too slow to wait for.
./run-dir.sh ~/aztec-repos/noir/test_programs/execution_success/!(regression_4709)
