# Noir Execute - A Noir Brillig to LLVM IR Transpiler

You know what nobody asked for? A Brillig to LLVM IR transpiler.

But people did ask for faster Brillig execution, and this was an experiment to explore if native code could achieve it.
The TLDR is that it does. Kind of. Sometimes. But for reasons largely independent of this approach, downsides, and we shouldn't use this.
We could take some learnings from this experiment and get most of the benefits in the rust Brillig execution VM with none of the downsides.

It also makes it quite clear that the real problem is poor Brillig bytecode generation in certain edge cases, rather than a slow execution engine.
The majority of interesting execution time for many programs will likely be in blackboxes, field arithmetic, foreign calls, oracles etc.

This _may_ have some use in the future, as some kind of AVM JIT transpiler for super-fast public function execution, if we're pushing limits.

# Implementation

- It uses rusts `inkwell` library to generate the LLVM IR.
- The calldata is embedded in the program, taken from `Prover.toml`.
- The program allocates the memory and the heap as a contiguous range of 256 bit "words".
- Field arithmetic will convert a field to montgomery form if it's not already, and track by setting msb to 1.
- Field arithmetic and Blackbox functions are implemented via calls into barretenberg c binds.
- It's probably not as optimised as it could be. But it does attempt to use the correct width operations for integer arithmetic.

On with some numbers.

# Measurements

This whole thing was kicked off when Mikes [blob-lib](https://github.com/iAmMichaelConnor/blob-lib) was revealed to take
"an hour to compile". By compile it's possible he meant "run a test", which combines compilation and execution.
Since then improvements have been made:

- Zac did some shenanigans to get it down to 15m.
- The Brillig VM had a bunch of memory allocations removed from the critical path.
- The Brillig codegen was improved to output 1/3rd less opcodes around array copies (in `aztec-packages` at least).

In aggregate these have got the `test_barycentric` run down to about 3m.
I modifed nargo to output the precise time executing so as to not conflate with compilation (which is pretty fast in all cases).
For native times with a `*` I subtracted 1500us (1.5ms) as that seems to be the overhead of running an empty program.

🤢 Implies we're far worse off in terms of time when factoring in compliation time.
In other words it would only provide a practical benefit if being re-executed over and over.

| Project         | Compressed Brillig | Uncompressed Brillig | `noir execute` time | x86 Time  | Speedup | x86 Compile Time | x86 Size Uncompressed |
| --------------- | ------------------ | -------------------- | ------------------- | --------- | ------- | ---------------- | --------------------- |
| blob-lib large  | 4,652,881          | 52,098,797 🤯        | 195s                | 48s       | 75%     | 203s 🤢          | 32,261,024            |
| blob-lib small  | 716,429            | 8,494,709 🤯         | 1.31s               | 0.39s     | 70%     | 9.8s 🤢          | 6,751,224             |
| 20m field mul   | 478                | 4191                 | 6.17s               | 1.63s     | 74%     | 7.619ms          | 58,352                |
| regression_5252 |                    |                      | 1.3s                | 931ms     | 30%     | 3.48s 🤢         |                       |
| cow_regression  |                    |                      | 10.48ms             | 1.572ms\* | 85%     | 99ms 🤢          |                       |

So as you can see above, when you factor in compilation time, blob-lib is actually slower than just executing normally.
However it is quite program dependent. The 20m field muls can compile and execute quite a bit faster than the rust vm.
This is largely due to using barretenberg field arithmetic and I think better handling of montgomery form.
Ostensibly there would be nothing stopping the rust vm achieving similar performance as well.
In other words, the actual program logic here (a tight loop) is negligible.

Further, this is with `-O0`, things only get far worse with higher levels, so much so I didn't bother waiting.
I may have waiting one time and it made no practical difference.

# Profiling

Running `blob-lib` large through `perf` we see:

- Brillig VM spent 42% of its time in malloc (still).
- Native code spent 30% of its time in `to_radix`.

We know we can get rid of heap allocs in the VM as the native code doesn't have any.
I'm unsure if the 30% time spent in `to_radix` can be improved upon, or if that much time is being spent in the rust vm.

# Usage

You can can meddle with the paths in `./run-tests.sh` to have it run a directory of projects (e.g. execution-success).
The paths to update would be to `aztec-packages/noir/noir-repo`, and the barretenberg branch `cl/spike-binds`.

Take a look at `./build.sh` for how to run a single project. It has a bunch of env vars you can set to control things.

| Env Var = Default | Description                                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------- |
| `O=0`             | The optimisation level passed to the compiler (currently only when with `ASM=1`).             |
| `BB=0`            | Wether to build barretenberg first. Useful when iterating on barretenberg code.               |
| `BB_DEBUG=0`      | Enable to use debug build of barretenberg.                                                    |
| `PACKAGES=0`      | Wether to attempt to install required packages. Assumes ubuntu.                               |
| `RUN=0`           | Enable to run the program through `time` after compiling.                                     |
| `NARGO=1`         | Disable to prevent recompilation of the noir project (unless no artifact exists).             |
| `CARGO=1`         | Disable to prevent recompilation of the transpiler (unless no exe exists).                    |
| `CARGO_DEBUG=0`   | Enable to use a debug build of the transpiler.                                                |
| `ASM=0`           | Enable to output intermediary `.ll` and `.s` files for inspection.                            |
| `AVX=0`           | Enable AVX on x86. Produces less bytecode, slightly more performant in theory.                |
| `VERBOSE=0`       | Enable to print every opcode with opcode, function and basic-block id.                        |
| `TRAP=0`          | Trigger an illegal instruction on failed assertions, rather than `exit(1)`. Helps debugging.  |
| `EXE=1`           | Link to create the executable when `ASM=1`. Disable if targetting another arch e.g. `risv64`. |

Example:

```
TRAP=1 AVX=1 ASM=1 RUN=1 ./build.sh ~/aztec-repos/aztec-packages/noir/noir-repo/test_programs/execution_success/1_mul
```

# SSA to LLVM IR

It maybe possible to get better bytecode by going direct from the SSA to the IR.
This would require building an SSA to IR code generator. The SSA and LLVM IR look quite similar, so this maybe quite neat.
This may however be no different to improving the SSA and Brillig codegen.

# Bugs

Probably. But it does run all the noir `execution-success` tests successfully (with a couple of uninteresting exceptions).
As per `./run-tests.sh`:

```
# regression_4709: Produces so much bytecode it's too slow to wait for.
# is_unconstrained: Fails because it expects main to be constrained, but we force all programs to be unconstrained.
# brillig_oracle: Requires advanced foreign function support to handle mocking.
# bigint: Might need to implement blackbox support. Although maybe bignum lib is the future.
```

# Conclusion

The result is not particularly helpful for us at this stage, and the outcome is somewhat unsurprising.
However it was a good way to learn the Brillig opcodes, VM implementation, and a bit about LLVM IR.

- We should focus on removing as many of the heap allocations as is reasonable from the rust VM critical path.
  I think the `ToRadix` handler might be what takes 40% of the time (or is at least contributing).
  This may matter less with other improvements to opcode generation, but any heap allocations should be unnecessary,
  perhaps with exception of some blackboxes.
- Opcode generation improvements. We already have work underway to reduce the amount of bytecode output, and this will
  make it easier to publish contracts on-chain, but won't reduce the execution trace.
  Mem2Reg improvements will hopefully make a big difference.
