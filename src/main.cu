
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <map>
#include <random>
#include <cmath>
#include <omp.h>
#define REAL float
#define TCSIZE 16
#define TCSQ 256
#define PRINTLIMIT 2560
#define WARPSIZE 32
#define DIFF (BSIZE<<3)

#define OMP_REDUCTION_FLOAT "omp-nt" STR(NPROC) "-float"
#define OMP_REDUCTION_DOUBLE "omp-nt" STR(NPROC) "-double"

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#include "nvmlPower.hpp"
#include "tools.cuh"
#include "kernel.cuh"
#include "variants.cuh"


int main(int argc, char **argv){
    // params
    if(argc != 8){
        fprintf(stderr, "run as ./prog dev n factor_ns seed REPEATS dist alg\nalg:\
        \n0 -> warp-shuffle\
        \n1 -> recurrence\
        \n2 -> single-pass\
        \n3 -> split\
        \n4 -> omp-reduction-float\
        \n5 -> omp-reduction-double\n\n");
        exit(EXIT_FAILURE);
    }
    int dev = atoi(argv[1]);
    long on = atoi(argv[2]);
    long n = on;
    float factor_ns = atof(argv[3]);
    int seed = atoi(argv[4]);
    int REPEATS = atoi(argv[5]);
    int dist = atoi(argv[6]);
    int alg = atoi(argv[7]);
    if(alg > 5){
        fprintf(stderr, "Error: Algorithms are in range 0, 1, 2, 3, 4, 5\n");
        exit(EXIT_FAILURE);
    }

#ifdef DEBUG
    const char* algorithms[6] = {"warp-shuffle", "recurrence", "single-pass", "split", OMP_REDUCTION_FLOAT, OMP_REDUCTION_DOUBLE};
    const char* disttext[3] = {"Normal Distribution", "Uniform Distribution", "Constant Distribution"};
    printf("\n\
            ***************************\n\
            dev            = %i\n\
            method         = %s\n\
            n              = %i\n\
            factor_ns      = %f\n\
            dist           = %s\n\
            prng_seed      = %i\n\
            REPEATS        = %i\n\
            TCSIZE         = %i\n\
            R              = %i\n\
            BSIZE          = %i\n\
            ***************************\n\n", dev, algorithms[alg], n, factor_ns, disttext[dist], seed, REPEATS, TCSIZE, R, BSIZE);
#endif
    
    // set device
    cudaSetDevice(dev);

    // mallocs
    #ifdef DEBUG 
        printf("CPU memory allocation        "); fflush(stdout);
    #endif
    REAL *A, *Ad;
    half *Adh, *outd_recA, *outd_recB;
    float *outd, *out;
    A = (REAL*)malloc(sizeof(REAL)*n);
    out = (float*)malloc(sizeof(float)*1);


    #ifdef DEBUG 
        printf("done\n"); fflush(stdout);
        printf("GPU memory allocation        "); fflush(stdout);
    #endif
    cudaMalloc(&Ad, sizeof(REAL)*n);
    cudaMalloc(&Adh, sizeof(half)*n);
    cudaMalloc(&outd, sizeof(float)*1);
    long smalln = (n + TCSQ-1)/TCSQ;
    //printf("small n = %lu   bs = %i\n", smalln, bs);
    cudaMalloc(&outd_recA, sizeof(half)*(smalln));
    cudaMalloc(&outd_recB, sizeof(half)*(smalln));

    #ifdef DEBUG 
        printf("done\n"); fflush(stdout);
        printf("Init data                    "); fflush(stdout);
    #endif
    init_distribution(A, n, seed, dist);

    #ifdef DEBUG 
        printf("done\n"); fflush(stdout);
        printf("Memcpy Host -> Dev           "); fflush(stdout);
    #endif
    cudaMemcpy(Ad, A, sizeof(REAL)*n, cudaMemcpyHostToDevice);
    #ifdef DEBUG 
        printf("done\n"); fflush(stdout);
        printf("[GPU] FP32 -> FP16           "); fflush(stdout);
    #endif
    convertFp32ToFp16 <<< (n + 256 - 1)/256, 256 >>> (Adh, Ad, n);
    cudaDeviceSynchronize();
    #ifdef DEBUG 
        printf("done\n"); fflush(stdout);
    #endif
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    #ifdef DEBUG
        //printf("%s (BSIZE = %i)...............", algorithms[alg], BSIZE); fflush(stdout);
        printf("[GPU] %-23s", algorithms[alg]); fflush(stdout);
    #endif
    cudaEventRecord(start);
    switch(alg){
        case 0:
            warpshuffle_reduction(Adh, outd, n, REPEATS);
            break;
        case 1:
            recurrence_reduction(Adh, outd, outd_recA, outd_recB, n, REPEATS);
            break;
        case 2:
            singlepass_reduction(Adh, outd, n, REPEATS);
            break;
        case 3:
            split_reduction(Adh, outd, n, factor_ns, REPEATS);
            break;
        case 4:
            omp_reduction<float>(A, out, n, REPEATS);
            break;
        case 5:
            omp_reduction<double>(A, out, n, REPEATS);
            break;
    }        
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    #ifdef DEBUG
        printf("done\n"); fflush(stdout);
    #endif
    if(alg<4){
        cudaMemcpy(out, outd, sizeof(float)*1, cudaMemcpyDeviceToHost);
    }

    
    float time = 0.0f;
    cudaEventElapsedTime(&time, start, stop);
    double goldtime = omp_get_wtime();
    #ifdef DEBUG 
        printf("[CPU] gold-reduction         "); fflush(stdout);
    #endif
    double cpusum = gold_reduction(A, n);
    #ifdef DEBUG 
        printf("done\n\n"); fflush(stdout);
    #endif


    goldtime = omp_get_wtime() - goldtime;
    #ifdef DEBUG
        printf("Benchmark Summary:\n\
        Alg:%-15s => %f (%f secs)\n\
        CPU gold reduction  => %f (%f secs)\n\
        Diff Result         = %f\n\
        Error               = %f%%\n\n", 
        algorithms[alg],
        (float)*out,
        time/(REPEATS*1000.0),
        (float)cpusum,
        goldtime,
        fabs((float)*out - cpusum),
        fabs(100.0f*fabs((float)*out - cpusum)/cpusum));
    #else
        printf("%f,%f,%f,%f,%f\n", time/(REPEATS), 
                (float)*out,cpusum,fabs((float)*out - cpusum),
                fabs(100.0f*fabs((float)*out - cpusum)/cpusum));
    #endif
    free(A);
    free(out);
    cudaFree(Ad);
    cudaFree(Adh);
    cudaFree(outd);
    cudaFree(outd_recA);
    cudaFree(outd_recB);
    exit(EXIT_SUCCESS);
}
