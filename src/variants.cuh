#pragma once

void warpshuffle_reduction(half *Adh, float *outd, long n, int REPEATS){
    dim3 block = dim3(BSIZE, 1, 1);
    dim3 grid = dim3((n + BSIZE -1)/BSIZE, 1, 1);
    #ifdef POWER
        GPUPowerBegin("warp-shuffle");
        #ifdef POWER_DEBUG
            printf("Measuring power consumption, press enter.....\n"); fflush(stdout);
            getchar();
        #endif
    #endif
    for(int i=0; i<REPEATS; ++i){
        cudaMemset(outd, 0, sizeof(REAL)*1);
        kernel_reduction_shuffle<<<grid, block>>>(Adh, outd, n);
        #ifdef DEBUG
            CUERR;
        #endif
        cudaDeviceSynchronize();
    }
    #ifdef POWER
        #ifdef POWER_DEBUG
            printf("done\n");
            getchar();
        #endif
        GPUPowerEnd();
    #endif
}

void recurrence_reduction(half *Adh, float *outd, half *outd_recA, half *outd_recB, long n, int REPEATS){
    dim3 block, grid;
    block = dim3(BSIZE, 1, 1);
    const int rlimit = 1;
    half *temp, resh;
    float resf;
    const int bs = BSIZE >> 5;
    #ifdef POWER
        GPUPowerBegin("recurrence");
        #ifdef POWER_DEBUG
            printf("Measuring power consumption, press enter.....\n"); fflush(stdout);
            getchar();
        #endif
    #endif
    for(int i=0; i<REPEATS; ++i){
        long dn = n;
        //grid = dim3((dn + TCSQ*bs-1)/(TCSQ*bs), 1, 1);
        grid = dim3((dn + (TCSQ*bs*(R)) - 1)/(TCSQ*bs*(R)), 1, 1);
        #ifdef DEBUG
            //printf("[kernel sample #%i]\n", i+1);
            //printf("dn=%12i >= rlimit =%5i  ", dn, rlimit);
            //printf("grid(%5i, %i, %i) block(%i, %i, %i).......", grid.x, grid.y, grid.z, block.x, block.y, block.z);
        #endif
        kernel_recurrence<<<grid, block>>>(Adh, outd_recA, dn);
        #ifdef DEBUG
            gpuErrchk( cudaPeekAtLastError() );
            gpuErrchk( cudaDeviceSynchronize() );
            //printf("done\n");
        #endif
        cudaDeviceSynchronize();
        //dn = (dn + TCSQ-1)/TCSQ;
        dn = (dn + TCSQ*(R)-1)/(TCSQ*(R));
        grid.x = (dn + (TCSQ*bs*(R)) - 1)/(TCSQ*bs*(R));
        while(dn > rlimit){
            #ifdef DEBUG
                //printf("dn=%12i >= rlimit =%5i  ", dn, rlimit);
                //printf("grid(%5i, %i, %i) block(%i, %i, %i).......", grid.x, grid.y, grid.z, block.x, block.y, block.z);
            #endif
            kernel_recurrence<<<grid, block>>>(outd_recA, outd_recB, dn);
            cudaDeviceSynchronize();
            //dn = (dn + TCSQ-1)/TCSQ;
            dn = (dn + TCSQ*(R)-1)/(TCSQ*(R));
            grid.x = (dn + (TCSQ*bs*(R)) - 1)/(TCSQ*bs*(R));
            #ifdef DEBUG
                gpuErrchk( cudaPeekAtLastError() );
                gpuErrchk( cudaDeviceSynchronize() );
                //printf("done\n");
            #endif
            temp = outd_recB;
            outd_recB = outd_recA;
            outd_recA = temp;
        }
        #ifdef DEBUG
            //printf("\n");
        #endif
    }
    #ifdef POWER
        #ifdef POWER_DEBUG
            printf("done\n");
            getchar();
        #endif
        GPUPowerEnd();
    #endif
    cudaMemcpy(&resh, outd_recA, sizeof(half), cudaMemcpyDeviceToHost);    
    resf = __half2float(resh); 
    cudaMemcpy(outd, &resf, sizeof(float), cudaMemcpyHostToDevice);    
}

void singlepass_reduction(half *Adh, float *outd, long n, int REPEATS){
    int bs = BSIZE >> 5;
    dim3 block = dim3(BSIZE, 1, 1);
    dim3 grid = dim3((n + (TCSQ*bs*(R)) - 1)/(TCSQ*bs*(R)), 1, 1);
    #ifdef KDEBUG
        printf("grid (%i, %i, %i)    block(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    #endif
    #ifdef POWER
        GPUPowerBegin("single-pass");
        #ifdef POWER_DEBUG
            printf("Measuring power consumption, press enter.....\n"); fflush(stdout);
            getchar();
        #endif
    #endif
    for(int i=0; i<REPEATS; ++i){
        cudaMemset(outd, 0, sizeof(REAL)*1);
        kernel_singlepass<<<grid, block>>>(Adh, outd, n, bs);
        cudaDeviceSynchronize();
    }
    #ifdef POWER
        #ifdef POWER_DEBUG
            printf("done\n");
            getchar();
        #endif
        GPUPowerEnd();
    #endif
}

// split approach: within a same block, some warps use Tensor Cores while others do shuffle
void split_reduction(half *Adh, float *outd, long n, float factor_ns, int REPEATS){
    int warps               = BSIZE >> 5;
    dim3 block              = dim3(BSIZE, 1, 1);
    int swarps             = (long)ceil(factor_ns*warps);
    int twarps             = warps - swarps;
    long datablock          = 256*twarps + 32*swarps;  
    int gridblocks          = (n + datablock-1)/datablock;
    dim3 grid               = dim3(gridblocks, 1, 1);
    #ifdef KDEBUG
        printf("BSIZE = %i    warps = %i    swarps %i    twarps %i   datablock %i\n", 
                BSIZE, warps, swarps, twarps, datablock);
        printf("grid (%i, %i, %i)    block(%i, %i, %i)  DIFF %i\n", grid.x, grid.y, grid.z, block.x, block.y, block.z,DIFF);
    #endif
    #ifdef POWER
        GPUPowerBegin("split");
        #ifdef POWER_DEBUG
            printf("Measuring power consumption, press enter.....\n"); fflush(stdout);
            getchar();
        #endif
    #endif
    for(int i=0; i<REPEATS; ++i){
        cudaMemset(outd, 0, sizeof(REAL)*1);
        kernel_split<<<grid, block>>>(n, Adh, outd, twarps, swarps, datablock);  CUERR;
        cudaDeviceSynchronize();
    }
    #ifdef POWER
        #ifdef POWER_DEBUG
            printf("done\n");
            getchar();
        #endif
        GPUPowerEnd();
    #endif
}

// original version, within the grid, some blocks do TC, others use CUDA cores.
void split_reduction_backup(half *Adh, float *outd, long n, float factor_ns, int REPEATS){
    int bs = BSIZE >> 5;
    dim3 block = dim3(BSIZE, 1, 1);
    long nsh = (long)ceil(factor_ns*n);
    long ntc = n - nsh;
    int ns_blocks = (nsh + BSIZE-1)/BSIZE;
    int tc_blocks = (ntc + TCSQ*bs - 1)/(TCSQ*bs);
    dim3 grid = dim3(tc_blocks + ns_blocks, 1, 1);
    #ifdef KDEBUG
        printf("ns_blocks %i, tc_blocks %i\n", ns_blocks, tc_blocks);
        printf("grid (%i, %i, %i)    block(%i, %i, %i)  DIFF %i\n", grid.x, grid.y, grid.z, block.x, block.y, block.z,DIFF);
    #endif
    #ifdef POWER
        GPUPowerBegin("split");
        #ifdef POWER_DEBUG
            printf("Measuring power consumption, press enter.....\n"); fflush(stdout);
            getchar();
        #endif
    #endif
    for(int i=0; i<REPEATS; ++i){
        cudaMemset(outd, 0, sizeof(REAL)*1);
        kernel_split_backup<<<grid, block>>>(n, Adh, outd, tc_blocks, ns_blocks);  CUERR;
        cudaDeviceSynchronize();
    }
    #ifdef POWER
        #ifdef POWER_DEBUG
            printf("done\n");
            getchar();
        #endif
        GPUPowerEnd();
    #endif
}



template<class T>
void omp_reduction(float *A, float *out, long n, int REPEATS){
    // variant 1: just one parallel region opened for all repetitions
    #ifdef POWER
        if(typeid(T) == typeid(float)){
            CPUPowerBegin(OMP_REDUCTION_FLOAT);
        }
        else{
            CPUPowerBegin(OMP_REDUCTION_DOUBLE);
        }
        #ifdef POWER_DEBUG
            printf("Measuring power consumption, press enter.....\n"); fflush(stdout);
            getchar();
        #endif
    #endif
    T acc;
    #pragma omp parallel shared(A,out, acc) num_threads(NPROC)
    {
        for(int k=0; k<REPEATS; ++k){
            #pragma omp single
            acc = (T)0.0f; 
            #pragma omp for schedule(static) reduction(+ : acc)
            for(int i = 0; i < n; ++i){
                acc += A[i];
            }
        }
    }
    *out = acc;
    #ifdef POWER
        #ifdef POWER_DEBUG
            printf("done\n");
            getchar();
        #endif
        CPUPowerEnd();
    #endif
}
