#include "cuda_md5.h"
#include "cuda_md5.cu"

__global__ void MD5Kernel(word64 *plaintext, word16 *md5Hash, unsigned int num_elements){
	
	const unsigned int tx = threadIdx.x;
	unsigned int index = tx + blockIdx.x*blockDim.x;
	unsigned int total_threads = blockDim.x*gridDim.x;
	
	uint8_t result[16];
	
	while (index < num_elements) {
		//memcpy(msg,plaintext[i], 64);
		cuda_md5((uint8_t*)plaintext[index].word, 56, result);
		memcpy(md5Hash[index].word, result, 16);
		index += total_threads;
	}
}
