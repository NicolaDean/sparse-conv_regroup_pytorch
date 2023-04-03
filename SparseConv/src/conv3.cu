#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define LENGTH 128
#define DEBUG 

extern "C" void spmm_conv(void *input_data_t, void *output_data_t, void *kernel_ptr_t, void *kernel_map_t, void *kernel_offset_t, void *kernel_data_t, void *kernel_ptr_sparse_t, void *kernel_map_sparse_t); 

static unsigned CudaTest(const char *msg) {
	
	cudaDeviceSynchronize();
	cudaError_t e = cudaGetLastError();
	if (cudaSuccess != (e)) {
		printf("\033[91m");
		printf("%s: %d\n", msg, e); 
		printf("%s in %s at line %d\n", cudaGetErrorString(e),__FILE__, __LINE__);
		printf("\033[0m");
		exit(-1);
		//return 1;
	}
	return 0;
}


#define CHECK_KERNELCALL()\
{ \
    const cudaError_t err = cudaGetLastError(); \
    if (err != cudaSuccess) { \
        printf("%s in %s at line %d\n", cudaGetErrorString(err),__FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    } \
}\

inline
cudaError_t checkCuda(cudaError_t result)
{
#if defined(DEBUG) || defined(_DEBUG)
  if (result != cudaSuccess) {
    fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(result));
    assert(result == cudaSuccess);
  }
#endif
  return result;
}

__global__
void _spmm_conv_0(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 23) + blockIdx.x * (23 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 23;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (1 * 23);
	int output_y = i /23 % 1;

	int x1 = output_x * 1 * 5 * 16 * 64 + output_y * 1 * 16 * 64 + c;

	float res[23<<1];
#pragma unroll
	for (int i=0; i<(23<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[23];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (16 * 5)  *5 * 16 * 64 + kernel_offset[threadIdx.x+b] / 16 % 5 * 16 * 64   + kernel_offset[threadIdx.x+b] % 16 * 64;
#pragma unroll
				for (int k=0; k<23; k++) {
					kernel_value[k] = kernel_data[threadIdx.x+b+length*k];
				}
			}
		}

		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+1);
		int idx3 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+2);
		int idx4 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+3);
		int idx5 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+4);
		int idx6 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+5);
		int idx7 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+6);
		int idx8 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+7);
#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (16 * 5)  *5 * 16 * 64 + kernel_offset[threadIdx.x+interm1] / 16 % 5 * 16 * 64   + kernel_offset[threadIdx.x+interm1] % 16 * 64;
#pragma unroll
			for (int k=0; k<23; k++) {
				kernel_value[k] = kernel_data[threadIdx.x+interm1+length*k];
			}
		}
	}

	if (interm1 < interm2) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+1);
		int idx3 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+2);
		int idx4 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+3);

#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<23; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*1*120+output_y*120)*64 + c;
#pragma unroll
	for (int k=0; k<23; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_1(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 22) + blockIdx.x * (22 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 22;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (1 * 22);
	int output_y = i /22 % 1;

	int x1 = output_x * 1 * 5 * 16 * 64 + output_y * 1 * 16 * 64 + c;

	float res[22<<1];
#pragma unroll
	for (int i=0; i<(22<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[22];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (16 * 5)  *5 * 16 * 64 + kernel_offset[threadIdx.x+b] / 16 % 5 * 16 * 64   + kernel_offset[threadIdx.x+b] % 16 * 64;
#pragma unroll
				for (int k=0; k<22; k++) {
					kernel_value[k] = kernel_data[threadIdx.x+b+length*k];
				}
			}
		}

		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+1);
		int idx3 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+2);
		int idx4 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+3);
		int idx5 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+4);
		int idx6 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+5);
		int idx7 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+6);
		int idx8 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+7);
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (16 * 5)  *5 * 16 * 64 + kernel_offset[threadIdx.x+interm1] / 16 % 5 * 16 * 64   + kernel_offset[threadIdx.x+interm1] % 16 * 64;
#pragma unroll
			for (int k=0; k<22; k++) {
				kernel_value[k] = kernel_data[threadIdx.x+interm1+length*k];
			}
		}
	}

	if (interm1 < interm2) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+1);
		int idx3 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+2);
		int idx4 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+3);

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*1*120+output_y*120)*64 + c;
#pragma unroll
	for (int k=0; k<22; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_2(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 22) + blockIdx.x * (22 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 22;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (1 * 22);
	int output_y = i /22 % 1;

	int x1 = output_x * 1 * 5 * 16 * 64 + output_y * 1 * 16 * 64 + c;

	float res[22<<1];
#pragma unroll
	for (int i=0; i<(22<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[22];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (16 * 5)  *5 * 16 * 64 + kernel_offset[threadIdx.x+b] / 16 % 5 * 16 * 64   + kernel_offset[threadIdx.x+b] % 16 * 64;
#pragma unroll
				for (int k=0; k<22; k++) {
					kernel_value[k] = kernel_data[threadIdx.x+b+length*k];
				}
			}
		}

		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+1);
		int idx3 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+2);
		int idx4 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+3);
		int idx5 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+4);
		int idx6 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+5);
		int idx7 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+6);
		int idx8 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+7);
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (16 * 5)  *5 * 16 * 64 + kernel_offset[threadIdx.x+interm1] / 16 % 5 * 16 * 64   + kernel_offset[threadIdx.x+interm1] % 16 * 64;
#pragma unroll
			for (int k=0; k<22; k++) {
				kernel_value[k] = kernel_data[threadIdx.x+interm1+length*k];
			}
		}
	}

	if (interm1 < interm2) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+1);
		int idx3 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+2);
		int idx4 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+3);

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<22; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*1*120+output_y*120)*64 + c;
#pragma unroll
	for (int k=0; k<22; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 


__global__
void _spmm_conv_3(const float * __restrict__ input_data, float *output_data, const int ptr_start, const int ptr_end, const int * __restrict__ kernel_ptr_all, const int * __restrict__ kernel_map_all, const int * __restrict__ kernel_offset, const float * __restrict__ kernel_data) {


	int i = (threadIdx.y * 17) + blockIdx.x * (17 << 2);
	int c = threadIdx.x + blockIdx.y * 64;

	const int *kernel_ptr = kernel_ptr_all + ptr_start;
	const int *kernel_map = kernel_map_all + ptr_start;

	int kernel_id = i % 17;
	int start = kernel_ptr[kernel_id];
	int end = kernel_ptr[kernel_id+1];
	int length = end - start;

	int output_x = i / (1 * 17);
	int output_y = i /17 % 1;

	int x1 = output_x * 1 * 5 * 16 * 64 + output_y * 1 * 16 * 64 + c;

	float res[17<<1];
#pragma unroll
	for (int i=0; i<(17<<1); i++) res[i] = 0.0f;

	int kernel_off;
	float kernel_value[17];
	int begin = 0;

	int interm1 = start + (((end - start) >> 3) << 3);
	int interm2 = start + (((end -start) >> 2) << 2);
	int interm3 = start + (((end -start) >> 1) << 1);


	for (int b=start; b<interm1; b+=8) {
		if (((b - start) & 31) == 0) {
			begin = b;
			if (threadIdx.x < end - b) {
				kernel_off = x1 + kernel_offset[threadIdx.x+b] / (16 * 5)  *5 * 16 * 64 + kernel_offset[threadIdx.x+b] / 16 % 5 * 16 * 64   + kernel_offset[threadIdx.x+b] % 16 * 64;
#pragma unroll
				for (int k=0; k<17; k++) {
					kernel_value[k] = kernel_data[threadIdx.x+b+length*k];
				}
			}
		}

		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+1);
		int idx3 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+2);
		int idx4 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+3);
		int idx5 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+4);
		int idx6 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+5);
		int idx7 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+6);
		int idx8 = __shfl_sync(0xFFFFFFFF, kernel_off, b-begin+7);
#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+4);
			res[k<<1] += val * input_data[idx5];
			res[(k<<1)+1] += val * input_data[idx5+32];
		}
#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+5);
			res[k<<1] += val * input_data[idx6];
			res[(k<<1)+1] += val * input_data[idx6+32];
		}

#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+6);
			res[k<<1] += val * input_data[idx7];
			res[(k<<1)+1] += val * input_data[idx7+32];
		}

#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], b-begin+7);
			res[k<<1] += val * input_data[idx8];
			res[(k<<1)+1] += val * input_data[idx8+32];
		}
	}
	

	
	if (interm1 < end && ((interm1-start)  & 31) == 0) {
		begin = interm1;
		if (threadIdx.x < end - interm1) {
			kernel_off = x1 + kernel_offset[threadIdx.x+interm1] / (16 * 5)  *5 * 16 * 64 + kernel_offset[threadIdx.x+interm1] / 16 % 5 * 16 * 64   + kernel_offset[threadIdx.x+interm1] % 16 * 64;
#pragma unroll
			for (int k=0; k<17; k++) {
				kernel_value[k] = kernel_data[threadIdx.x+interm1+length*k];
			}
		}
	}

	if (interm1 < interm2) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+1);
		int idx3 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+2);
		int idx4 = __shfl_sync(0xFFFFFFFF, kernel_off, interm1-begin+3);

#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin);
			res[k<<1] += val>0? val * input_data[idx]:0;
			res[(k<<1)+1] += val>0? val * input_data[idx+32]:0;
		}

#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+2);
			res[k<<1] += val * input_data[idx3];
			res[(k<<1)+1] += val * input_data[idx3+32];
		}
#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm1-begin+3);
			res[k<<1] += val * input_data[idx4];
			res[(k<<1)+1] += val * input_data[idx4+32];
		}
	}

	if (interm2 < interm3) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin);
		int idx2 = __shfl_sync(0xFFFFFFFF, kernel_off, interm2-begin+1);

#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm2-begin+1);
			res[k<<1] += val * input_data[idx2];
			res[(k<<1)+1] += val * input_data[idx2+32];
		}
	}
		
	if (interm3 < length) {
		int idx = __shfl_sync(0xFFFFFFFF, kernel_off, interm3-begin);
#pragma unroll
		for (int k=0; k<17; k++) {
			float val = __shfl_sync(0xFFFFFFFF, kernel_value[k], interm3-begin);
			res[k<<1] += val * input_data[idx];
			res[(k<<1)+1] += val * input_data[idx+32];
		}
	}

	int output_idx = (output_x*1*120+output_y*120)*64 + c;
#pragma unroll
	for (int k=0; k<17; k++) {
		output_data[output_idx+kernel_map[kernel_id+k]*64] = res[k<<1];
		output_data[output_idx+kernel_map[kernel_id+k]*64+32] = res[(k<<1)+1];
	}

} 




void spmm_conv(void *input_data_t, void *output_data_t, void *kernel_ptr_t, void *kernel_map_t, void *kernel_offset_t, void *kernel_data_t, void *kernel_ptr_sparse_t, void *kernel_map_sparse_t) {
	float *input_data = (float *)input_data_t;
	float *output_data = (float *)output_data_t;
	int *kernel_ptr = (int *)kernel_ptr_t;
	int *kernel_map = (int *)kernel_map_t;
	int *kernel_offset = (int *)kernel_offset_t;
	float *kernel_data = (float *)kernel_data_t;
	int *kernel_ptr_sparse = (int *)kernel_ptr_sparse_t;
	int *kernel_map_sparse = (int *)kernel_map_sparse_t;

	cudaStream_t stream_0;
cudaStream_t stream_1;
cudaStream_t stream_2;
cudaStream_t stream_3;


	cudaStreamCreate(&stream_0);
dim3 nblocks_0(1, 1);
dim3 nthreads_0(32, 4);
_spmm_conv_0<<<nblocks_0, nthreads_0, 0, stream_0>>>(input_data, output_data, 0, 24, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_1);
dim3 nblocks_1(1, 1);
dim3 nthreads_1(32, 4);
_spmm_conv_1<<<nblocks_1, nthreads_1, 0, stream_1>>>(input_data, output_data, 24, 47, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_2);
dim3 nblocks_2(1, 1);
dim3 nthreads_2(32, 4);
_spmm_conv_2<<<nblocks_2, nthreads_2, 0, stream_2>>>(input_data, output_data, 47, 70, kernel_ptr, kernel_map, kernel_offset, kernel_data);
cudaStreamCreate(&stream_3);
dim3 nblocks_3(1, 1);
dim3 nthreads_3(32, 4);
_spmm_conv_3<<<nblocks_3, nthreads_3, 0, stream_3>>>(input_data, output_data, 70, 88, kernel_ptr, kernel_map, kernel_offset, kernel_data);


	CudaTest("Something gone wrong");

	cudaStreamDestroy(stream_0);
cudaStreamDestroy(stream_1);
cudaStreamDestroy(stream_2);
cudaStreamDestroy(stream_3);

}


