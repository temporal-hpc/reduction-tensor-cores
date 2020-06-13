#pragma once

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

//#define DEBUG

void init_normal(REAL *m, long n, const float mean, float var, int seed){
    //srand(seed);
    std::mt19937 gen{seed};
    std::normal_distribution<> d{mean,var};
    for(long k=0; k<n; ++k){
        m[k] = (float) d(gen); //(REAL) rand()/(((REAL)RAND_MAX)*1000);
        //printf("%f\n",(float)m[k]);
    }
}

void init_uniform(REAL *m, long n, const float low, const float high, int seed){
    std::mt19937 gen{seed};
    std::uniform_real_distribution<> d(0, 1);
    for(long k=0; k<n; ++k){
        m[k] = (float) d(gen);//(REAL) rand()/(((REAL)RAND_MAX)*1000);
        //printf("%f\n",(float)m[k]);
    }
}

void init_constant(REAL *m, long n, const float val, int seed){
    for(long k=0; k<n; ++k){
        m[k] = (float) val;
    }
}

void init_distribution(REAL *m, long n, int seed, int dist){
    switch(dist){
        case 0: 
            init_normal(m, n, 0.0f, 1.0f, seed);
            break;
        case 1:
            init_uniform(m, n, 0.0f, 1.0f, seed);
            break;
        case 2:
            init_constant(m, n, 0.01f, seed);
            break;
        default:
            init_normal(m, n, 0.0f, 1.0f, seed);
            break;
    }
}



double gold_reduction(REAL *m, long n){
    double sum = 0.0f;
    for(long k=0; k<n; ++k){
        sum += m[k];
    }
    return sum;
}

void printarray(REAL *m, int n, const char *msg){
    printf("%s:\n", msg);
    for(int j=0; j<n; ++j){
        printf("%8.4f\n", m[j]);
    }
}

void printmats(REAL *m, int n, const char *msg){
    long nmats = (n + 256 - 1)/TCSQ;
    printf("%s:\n", msg);
    long index=0;
    for(int k=0; k<nmats; ++k){
        printf("k=%i\n", k);
        int off = k*TCSIZE*TCSIZE;
        for(int i=0; i<TCSIZE; ++i){
            for(int j=0; j<TCSIZE; ++j){
                if(index < n){
                    printf("%8.4f ", m[off + i*TCSIZE + j]);
                }
                else{
                    printf("%8.4f ", -1.0f);
                }
                index += 1;
            }
            printf("\n");
        }
    }
}

