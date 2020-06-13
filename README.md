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
### d) Other compile options
Choose GPU arch: ARCH=<sm arch>
Choose OpenMP threads (just for OpenMP mode) NPROCS=<value>

## Run
run as ./prog dev n factor_ns seed REPEATS dist alg
alg:        
0 -> warp-shuffle        
1 -> recurrence        
2 -> single-pass        
3 -> split        
4 -> omp-reduction-float        
5 -> omp-reduction-double



