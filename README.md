# GPU Accelerated Arithmetic Reduction with Tensor Cores
Fast GPU based reductions using tensor cores

## Compile

### a) With visual progress
``` 
make DEBUG=DEBUG 
```
### b) With power and energy metrics
``` 
make DEBUG=DEBUG POWER=POWER 
```
### c) Just for benchmark
```
make
```
### d) Other compile options (see Makefile)
```
GPU arch        : ARCH
Block Size      : BSIZE
R Chain         : R
OpenMP threads  : NPROCS
```

## Run
run as 
```
./bin/prog dev n factor_ns seed REPEATS dist alg

dev         = GPU ID (i.e., 0, 1, 2, ...)
factor_ns   = Fraction of `n` to be processed by [REF] Warp Shuffle
seed        = PRNG seed for number generation
REPEATS     = Number of kernel execution repeats (for more stable average time measurements)
dist        = Data distribution: 0 -> Normal,  1 -> Uniform,  2 -> Constant
alg:        
0 -> [REF] warp-shuffle        
1 -> [Proposed Variant] recurrence        
2 -> [Proposed Variant] single-pass        
3 -> [Proposed Variant] split        
4 -> omp-reduction-float        
5 -> omp-reduction-double
```
