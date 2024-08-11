#!/bin/bash
shopt -s extglob

# regression_4709: Produces so much bytecode it's too slow to wait for.
# is_unconstrained: Fails because it expects main to be constrained, but we force all programs to be unconstrained.
# brillig_oracle: Requires advanced foreign function support to handle mocking.
# bigint: Might need to implement blackbox support. Although maybe bignum lib is the future.
./run-dir.sh ~/aztec-repos/noir/test_programs/execution_success/!(regression_4709|is_unconstrained|brillig_oracle|bigint)
