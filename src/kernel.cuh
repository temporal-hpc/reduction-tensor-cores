#ifndef _KERNEL_H
#define _KERNEL_H_
using namespace nvcuda;
//using namespace cooperative_groups;
// The only dimensions currently supported by WMMA
const int WMMA_M = 16;
const int WMMA_N = 16;
const int WMMA_K = 16;

#define CUERR {                                                            \
    cudaError_t err;                                                       \
    if ((err = cudaGetLastError()) != cudaSuccess) {                       \
        printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err),__FILE__, __LINE__); \
        exit(1);                                                           \
    }                                                                      \
}
//printf("0, 0, 0, 0, 0\n", cudaGetErrorString(err), __FILE__, __LINE__); \
//printf("CUDA error: %s : %s, line %d\n", cudaGetErrorString(err), __FILE__, __LINE__);
__global__ void convertFp32ToFp16 (half *out, float *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}
__global__ void convertFp16ToFp32 (float *out, half *in, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}
__inline__ __device__ half warp_shuffle_reduction_half(half val){
	for (int offset = WARPSIZE >> 1; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFF, val, offset, WARPSIZE);
    return val;
}

__inline__ __device__ REAL warp_shuffle_reduction_real(REAL val){
	for (int offset = WARPSIZE >> 1; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset, WARPSIZE);
    return val;
}

// kernel
//__global__ void tc_reduction(half* A, int n){
__inline__ __device__ REAL reduction_tc_warp(int N, half *A, int offset, int lane, int warpoff){
    // definicion de offset y fragmentos
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, REAL> r_frag;
    
    // (1) cargar datos de memoria global a A, B y C frags
    wmma::fill_fragment(a_frag, 1.0f);
    //wmma::fill_fragment(b_frag, 0.0f);
    wmma::fill_fragment(d_frag, 0.0f);

    // (2) mejora MMA multiples encadenados
    //const int bigoffset = gridDim.x * 32 * TCSQ;
    //if(offset >= N){ return 0.0f; }
    #pragma loop unroll
    for(int i=0; i<R; ++i){
        //if(threadIdx.x == 0) printf("offset %i \n",offset + TCSQ*32*(i+1));
        wmma::load_matrix_sync(b_frag, A + offset + (i<<8), TCSIZE);
        wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);
    }

    // (3) preparando datos para segundo MMA
    wmma::fill_fragment(b_frag, 1.0f);
    // [OPCION 1] copia manual
    //#pragma loop unroll
    //for(int i=0; i < 8 ; ++i){
    //    a_frag.x[i] = d_frag.x[i];
    //    a_frag.x[i+8] = d_frag.x[i];
    //}
   
    //int offwid = (threadIdx.x/32)*256;
    // [OPCION 2] copia a shared mem
    __shared__ half As[DIFF];
    wmma::store_matrix_sync(As+warpoff, d_frag, TCSIZE, wmma::mem_row_major);
    wmma::load_matrix_sync(a_frag, As+warpoff, TCSIZE);
    wmma::fill_fragment(d_frag, 0.0f);




    //// (4) MMA
    wmma::mma_sync(r_frag, a_frag, b_frag, d_frag);

    // (5) Almacenar resultado
    if(lane == 0){
        //printf("block: %i, val %f\n",blockIdx.x,(float)d_frag.x[0]);
        //printf("%f\n",(float)d_frag.x[0]);
        return r_frag.x[0];
        //return 1.0f;
    }
    else return 0.0f;
}

// kernel
//__global__ void tc_reduction(half* A, int n){
__inline__ __device__ REAL reduction_tc_warp_R1(int N, half *A, int offset, int lane, int warpoff){
    // definicion de offset y fragmentos
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, REAL> r_frag;
    
    // (1) cargar datos de memoria global a A, B y C frags
    wmma::fill_fragment(a_frag, 1.0f);
    //wmma::fill_fragment(b_frag, 0.0f);
    wmma::fill_fragment(d_frag, 0.0f);

    // (2) mejora MMA multiples encadenados
    //const int bigoffset = gridDim.x * 32 * TCSQ;
    //if(offset >= N){ return 0.0f; }
    //if(threadIdx.x == 0) printf("offset %i \n",offset + TCSQ*32*(i+1));
    wmma::load_matrix_sync(b_frag, A + offset, TCSIZE);
    wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);

    // (3) preparando datos para segundo MMA
    wmma::fill_fragment(b_frag, 1.0f);
    // [OPCION 1] copia manual
    //#pragma loop unroll
    //for(int i=0; i < 8 ; ++i){
    //    a_frag.x[i] = d_frag.x[i];
    //    a_frag.x[i+8] = d_frag.x[i];
    //}
   
    //int offwid = (threadIdx.x/32)*256;
    // [OPCION 2] copia a shared mem
    __shared__ half As[DIFF];
    wmma::store_matrix_sync(As+warpoff, d_frag, TCSIZE, wmma::mem_row_major);
    wmma::load_matrix_sync(a_frag, As+warpoff, TCSIZE);
    wmma::fill_fragment(d_frag, 0.0f);




    //// (4) MMA
    wmma::mma_sync(r_frag, a_frag, b_frag, d_frag);

    // (5) Almacenar resultado
    if(lane == 0){
        //printf("block: %i, val %f\n",blockIdx.x,(float)d_frag.x[0]);
        //printf("%f\n",(float)d_frag.x[0]);
        return r_frag.x[0];
        //return 1.0f;
    }
    else return 0.0f;
}

__inline__ __device__ REAL block_reduce_tc(int N, half *a, int offset){
	__shared__ REAL shared[WARPSIZE];
	int tid = threadIdx.x;
	int lane = tid & (WARPSIZE-1);
	//int wid = tid/WARPSIZE;
	int wid = tid >> 5;
	REAL val = reduction_tc_warp(N, a, offset + wid*TCSQ*R, lane, wid << 8);
    //return val;
	if(lane == 0){
		shared[wid] = val;
    }
	__syncthreads();
	//val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : (REAL) 0.0f;
    //printf("thread %i val %f\n", threadIdx.x, val);
	val = (tid < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
	if(wid == 0){
        val = warp_shuffle_reduction_real(val);
    }
	return val;
}

 __inline__ __device__ half block_reduce_shuffle(half val){
     __shared__ half shared[WARPSIZE];
     int tid = threadIdx.x;
     int lane = tid & (WARPSIZE-1);
     int wid = tid/WARPSIZE;
     val = warp_shuffle_reduction_half(val);
     if(lane == 0)
         shared[wid] = val;
 
     __syncthreads();
     val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : (half) 0.0f;
     if(wid == 0){
        val = warp_shuffle_reduction_real(val);
     }
     return val;
 }







__global__ void kernel_singlepass(half *a, float *out, int N,int bs){
	//int offset = blockIdx.x * TCSQ * 32;       
	int offset = blockIdx.x * (bs * TCSQ * R); 
	if(offset < N){
		REAL sumf = block_reduce_tc(N, a, offset);
        //if((threadIdx.x & 31) == 0){
        //    //printf("offset %i \n",offset);
        //    atomicAdd(out, sumf);
        //}
        if(threadIdx.x == 0){
            //printf("offset %i \n",offset);
            atomicAdd(out, sumf);
        }
	}
}




//////////////////////
// SPLIT
//////////////////////
__inline__ __device__ REAL block_reduce_tc_R1(int N, half *a, int offset){
	__shared__ REAL shared[WARPSIZE];
	int tid = threadIdx.x;
	int lane = tid & (WARPSIZE-1);
	//int wid = tid/WARPSIZE;
	int wid = tid >> 5;
	REAL val = reduction_tc_warp_R1(N, a, offset + wid*TCSQ, lane, wid << 8);
    //return val;
	if(lane == 0){
		shared[wid] = val;
    }
	__syncthreads();
	//val = (tid < blockDim.x/WARPSIZE) ? shared[lane] : (REAL) 0.0f;
    //printf("thread %i val %f\n", threadIdx.x, val);
	val = (tid < (blockDim.x >> 5)) ? shared[lane] : 0.0f;
	if(wid == 0){
        val = warp_shuffle_reduction_real(val);
    }
	return val;
}

// [ TC MMAs ..... |  CLASSIC SHUFFLE ]
__global__ void kernel_split(int N, half *a, float *out, int bntc, int bns){
    REAL sum=0;
    int offset_tc = (blockIdx.x)*DIFF;
    if(blockIdx.x < bntc){
        sum = block_reduce_tc_R1(N, a, offset_tc);
    }
    else{
    //if(blockIdx.x >= bntc){
        int offset_shuffle = (blockIdx.x-bntc) * blockDim.x + threadIdx.x;
        sum = block_reduce_shuffle(a[offset_shuffle+bntc*DIFF]);
        //if(threadIdx.x==0)printf("%f\n",(float)offset_shuffle);
    }
/*    if(offset_shuffle < bns*BSIZE){
        sum = block_reduce_shuffle(a[offset_shuffle]);
    }
    else{
        int offset_tc = (blockIdx.x)*DIFF+bns*BSIZE;
        sum = block_reduce_tc(N, a, offset_tc);
        //int offset_tc = (blockIdx.x - bns) * DIFF;
        //sum = block_reduce_tc(a, bns*BSIZE+offset_tc);
    }
  */  if(threadIdx.x == 0){
        atomicAdd(out, sum);
    }
}













// backup kernel assuming R=1 (this one is not being used)
__global__ void kernel_recurrence_R1(half* in, half* out, long n){
    __shared__ half smat[TCSQ];
    __shared__ half As[DIFF];
    const half hz = 0.0f;
    int wid = threadIdx.x >> 5;
    int offwid = wid << 8;
    int wlane = threadIdx.x & 31;
    long warp_offset = blockIdx.x*DIFF + offwid;
    int gwid = (BSIZE >> 5)*blockIdx.x + wid;  // global warp id
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    wmma::fill_fragment(a_frag, 1.0f);
    wmma::fill_fragment(c_frag, 0.0f);

    if(warp_offset >= n){ return; }
    if(warp_offset + TCSQ <= n){
        wmma::load_matrix_sync(b_frag, in + warp_offset, TCSIZE);
    }
    else{
        // last warp (part within n, part over n)
        #pragma loop unroll
        for(int i=0; i<8; ++i){
            smat[wlane + (i << 5)] = (warp_offset + wlane + (i<<5)) < n ? in[warp_offset + wlane + (i<<5)] : hz;
        }
        // don't need __syncthreads as it is just one warp working on SM
        wmma::load_matrix_sync(b_frag, smat, TCSIZE);
    }
    // (3) MMA #1
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
    wmma::fill_fragment(b_frag, 1.0f);
         
    wmma::store_matrix_sync(As+offwid, d_frag, TCSIZE, wmma::mem_row_major);
    wmma::load_matrix_sync(a_frag, As+offwid, TCSIZE);
    //wmma::fill_fragment(d_frag, 0.0f);
             
    // (4) MMA #2
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
 
    // (5) Store per-warp result, warps that act on locations greater than n do not reach this line
    if(wlane == 0){
        out[gwid] = d_frag.x[0];
        #ifdef KDEBUG
            printf("\n[gwid=%i] [wid=%i] [wlane=%i] [Bid = %i] result = %f\n", gwid, wid, wlane, blockIdx.x, (float) out[gwid]);
        #endif
    }
}

__global__ void kernel_recurrence(half* in, half* out, long n){
    __shared__ half smat[TCSQ];
    __shared__ half As[DIFF];
    //__shared__ half row1[TCSQ];
    //__shared__ half col1[TCSQ];
    const half hz = 0.0f;
    int wid = threadIdx.x >> 5;
    int offwid = wid << 8;
    int wlane = threadIdx.x & 31;
    long warp_offset = R*(blockIdx.x*DIFF + offwid);
    int gwid = (BSIZE >> 5)*blockIdx.x + wid;  // global warp id
    int woffR;

    /*
    // row1 col1 approach
    if(wid == 0){
        if(threadIdx.x < 16){
            row1[threadIdx.x] = 1.0f;
            #pragma unroll
            for(int i=1; i<TCSIZE; ++i){
                row1[TCSIZE*i + threadIdx.x] = 0.0f;
            }
        }
        else{
            col1[TCSIZE*(threadIdx.x - TCSIZE)] = 1.0f;
            #pragma unroll
            for(int j=1; j<TCSIZE; ++j){
                col1[TCSIZE*(threadIdx.x-TCSIZE) + j] = 0.0f;
            }
        }
    }
    __syncthreads();
    */
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> d_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, half> c_frag;
    // full 1s approach
    wmma::fill_fragment(a_frag, 1.0f);
    // row1 col1 approach
    //wmma::load_matrix_sync(a_frag, row1, TCSIZE);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::fill_fragment(d_frag, 0.0f);

    if(warp_offset >= n){ return; }
    for(int r=0; r<R; ++r){
        woffR = warp_offset + TCSQ*r;
        if(woffR >= n){ break; }
        if(woffR + TCSQ <= n){
            wmma::load_matrix_sync(b_frag, in + woffR, TCSIZE);
        }
        else{
            // last warp (part within n, part over n)
            #pragma loop unroll
            for(int i=0; i<8; ++i){
                smat[wlane + (i << 5)] = (woffR + wlane + (i<<5)) < n ? in[woffR + wlane + (i<<5)] : hz;
            }
            // don't need __syncthreads as it is just one warp working on smat
            wmma::load_matrix_sync(b_frag, smat, TCSIZE);
        }
        // (3) MMA #1
        wmma::mma_sync(d_frag, a_frag, b_frag, d_frag);
    }

    // full 1s approach
    wmma::fill_fragment(b_frag, 1.0f);
    // row1 col1
    //wmma::load_matrix_sync(b_frag, col1, TCSIZE);
    wmma::store_matrix_sync(As+offwid, d_frag, TCSIZE, wmma::mem_row_major);
    wmma::load_matrix_sync(a_frag, As+offwid, TCSIZE);
    //wmma::fill_fragment(d_frag, 0.0f);
             
    // (4) MMA #2
    wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
 
    // (5) Store per-warp result, warps that act on locations greater than n do not reach this line
    if(wlane == 0){
        out[gwid] = d_frag.x[0];
        #ifdef KDEBUG
            printf("\n[gwid=%i] [wid=%i] [wlane=%i] [Bid = %i] result = %f\n", gwid, wid, wlane, blockIdx.x, (float) out[gwid]);
        #endif
    }
}



__global__ void kernel_reduction_shuffle(half *A, float *out, int n){
     int off = blockIdx.x * blockDim.x + threadIdx.x;
     if(off < n){
         half sum = A[off];
         float rsum = block_reduce_shuffle(sum);
         if(threadIdx.x == 0){
             atomicAdd(out, rsum);
         }
     }
 }
#endif
