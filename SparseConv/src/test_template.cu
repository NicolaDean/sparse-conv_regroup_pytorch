#include <stdio.h>
#include <assert.h>
#include <cuda.h>

#define LENGTH 128
#define DEBUG 

extern "C" void spmm_conv(void *input_data_t, void *output_data_t, void *kernel_ptr_t, void *kernel_map_t, void *kernel_offset_t, void *kernel_data_t, void *kernel_ptr_sparse_t, void *kernel_map_sparse_t); 

static unsigned CudaTest(const char *msg) {
	cudaError_t e;
	cudaDeviceSynchronize();
	if (cudaSuccess != (e = cudaGetLastError())) {
		fprintf(stderr, "%s: %d\n", msg, e); 
		fprintf(stderr, "%s\n", cudaGetErrorString(e));
		exit(-1);
		//return 1;
	}
	return 0;
}


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

_CODE_KERNEL

void spmm_conv(void *input_data_t, void *output_data_t, void *kernel_ptr_t, void *kernel_map_t, void *kernel_offset_t, void *kernel_data_t, void *kernel_ptr_sparse_t, void *kernel_map_sparse_t) {
	float *input_data = (float *)input_data_t;
	float *output_data = (float *)output_data_t;
	int *kernel_ptr = (int *)kernel_ptr_t;
	int *kernel_map = (int *)kernel_map_t;
	int *kernel_offset = (int *)kernel_offset_t;
	float *kernel_data = (float *)kernel_data_t;
	int *kernel_ptr_sparse = (int *)kernel_ptr_sparse_t;
	int *kernel_map_sparse = (int *)kernel_map_sparse_t;

	_DECL_STREAM

	float time;
	
	_CALL_KERNEL

	CudaTest("Something gone wrong");

	_CLEAN_UP
}


