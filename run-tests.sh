#!/bin/bash

shopt -s extglob

./run_dir.sh ~/aztec-repos/noir/test_programs/execution_success/!(regression_4709)
