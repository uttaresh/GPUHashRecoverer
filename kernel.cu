#include "cuda_md5.h"
#include "cuda_md5.cu"

__constant__ uint8_t intTest_d[16];
__constant__ password seq_d[20];
__constant__ char chars_d[NUM_LETTERS];

__device__ inline int cmpResult(uint8_t res1[], uint8_t res2[]){
	int j=0;
	int flag=0;
	while(j<16){
		if(res1[j] != res2[j]){
               flag = 1;
		}
		++j;
	}
	return flag;
}

__device__ inline void getModifiedWord(int z, password *w, password *oldword){

	memcpy(w,oldword, sizeof(password));

	if(z == -1){
	} else if (z==0){
		/* First letter uppercase */
		if ((*w).word[0] >= 'a' && (*w).word[0] <= 'z')
		    (*w).word[0] +='A'-'a';
	} else if (z==1){
		/* Last letter uppercase */
		size_t len = (*w).length;
		if ((*w).word[len-1] >= 'a' && (*w).word[len-1] <= 'z')
		    (*w).word[len-1] += 'A'-'a';
	} else if (z>=2 && z<=11){
		/* Add one digit to end
		 * iterator: z-2    */
		size_t len = (*w).length;
		(*w).word[len] = '0' + z-2;
		(*w).length += 1;
	} else if (z>=12 && z<=111){
		/* Add sequence of numbers at end; e.g. 1234, 84, 1999 */
		// 0 to 99
		// iterator: z-12
		size_t len = (*w).length;
		(*w).word[len] = '0' + ((z-12)/10)%10;
		(*w).word[len+1] = '0' + (z-12)%10;
		(*w).length += 2;
	} else if (z>=112 && z<=231){
		// 1900 to 2020
		// iterator: z + (1900-112)
		size_t len = (*w).length;
		(*w).word[len] = '0' + ((z+1900-112)/1000)%10;
		(*w).word[len+1] = '0' + ((z+1900-112)/100)%10;
		(*w).word[len+2] = '0' + ((z+1900-112)/10)%10;
		(*w).word[len+3] = '0' + (z+1900-112)%10;
		(*w).length += 4;
	} else if (z>=232 && z<=251){
		// Other common sequences
		// iterator: z-232
		//sprintf(&temp,"%s",sequences[z-252]);
		size_t len = (*w).length;
		memcpy(&((*w).word[len]),seq_d[z-232].word,seq_d[z-232].length);
		(*w).length = len + seq_d[z-232].length;
	}
}

/* Brute force password recovery */
__global__ void BruteKernel(password *matchPwd, unsigned int pass_length, uint64_t words_per_thread, uint64_t total_words, volatile bool *kernelFound) {	

	unsigned int tx = threadIdx.x + blockIdx.x * blockDim.x; // overall thread #
	uint64_t start_word = tx*words_per_thread; // the starting index of password this thread checks

	/* initialize variables */
	uint64_t k = 0;
	unsigned int i = 0;
	uint64_t curr_word = start_word;
	password pwd;
	
	volatile __shared__ bool foundInBlock;
	uint8_t result[16];	
	
    if (tx == 0) foundInBlock = *kernelFound;
    __syncthreads();

	/* Each thread checks all the passwords it's assigned to */
	while((!foundInBlock) && (k < words_per_thread) && (curr_word < total_words)) {
	
		memset(&pwd,0,sizeof(password));
		pwd.length = pass_length;
		/* construct the curr_word #'s password */
		for(i=0; i<pass_length; i++) {
			pwd.word[pass_length-1-i] = chars_d[(curr_word % NUM_LETTERS)];
			curr_word /= NUM_LETTERS;
		}
		
		cuda_md5(&pwd,result);
		if(cmpResult(result, intTest_d) == 0){
			memcpy(matchPwd,&pwd,sizeof(password));
			foundInBlock = true;
			*kernelFound = true;
		}
		//To reduce the global memory traffic
		++curr_word;
		++k;
		if (threadIdx.x == 0 && *kernelFound) foundInBlock = true;
		//__syncthreads();
	}
}

__global__ void DictKernel(password *pwd, password *matchPwd, volatile bool *kernelFound, unsigned int num_elements){
	
	unsigned int tx = threadIdx.x;
	unsigned int index = tx + blockIdx.x*blockDim.x;
	unsigned int total_threads = blockDim.x*gridDim.x;
	
   	volatile __shared__ bool foundInBlock;
	
	uint8_t result[16];
	password w;
	
   	// initialize shared status
    if (tx == 0) foundInBlock = *kernelFound;
    __syncthreads();
   	
   	
	while (!foundInBlock && index < num_elements) {		
		for(int z=-1; z<252; z++){
			getModifiedWord(z, &w, &(pwd[index]));
			cuda_md5(&w,result);

			if(cmpResult(result, intTest_d) == 0){
				memcpy(matchPwd,&w,sizeof(password));
				foundInBlock = true;
				*kernelFound = true;
			}
		}
		
		index += total_threads;
		
		//To reduce the global memory traffic
		if (threadIdx.x == 0 && *kernelFound) foundInBlock = true;
		//__syncthreads();
	}
}
