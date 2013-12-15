#include "support.cu"
#include "kernel.cu"

void readPwdFromFile(FILE *infile, char ***pwd, unsigned int *numLines){

	unsigned int numberOfLines = 0;
	int ch;
	while (EOF != (ch=getc(infile))){
    	if (ch=='\n'){
    		++numberOfLines;
	    	if(numberOfLines == (UINT_MAX/sizeof(char*)))
	    		break;
    	}
    }
    rewind(infile);

    *pwd = (char **)malloc(numberOfLines * sizeof(char*));

    char *line = NULL;
	size_t len = 0;
	int read_len = 0;
	unsigned int i=0;
	unsigned int toReduce = 0;
	while (i<numberOfLines) {
		read_len = getline(&line, &len, infile);
		if(read_len != -1){
			if(line[read_len-1] == '\n'){
                line[read_len-1] = '\0';
                read_len = read_len - 1;
            }
			if(line[read_len-1] == '\r'){
			    line[read_len-1] = '\0';
				read_len = read_len - 1;
            }
			if((read_len+1) > 55){
                //printf("%s is too big. Skipping.\n",line);
                ++toReduce;
            } else {
                (*pwd)[i-toReduce] = (char*)malloc( (read_len+1)*sizeof(char));
                memcpy((*pwd)[i-toReduce],line,read_len+1);
                //printf("Pwd Read: %s, %d\n", (*pwd)[i], read_len);
	  		}
	  	} else {
	  		break;
	  	}
		free(line);
		line = NULL;
		len = 0;
	  	i++;
	}
	*numLines = numberOfLines-toReduce;
	//passwd = &pwd;
}

void preProcessPwd(char **pwd, word64 *plaintext, unsigned int num){

	unsigned int i=0;
	while(i<num) {
        //printf("Preprocessing - %s\n",pwd[i]);
		unsigned int str_len = (unsigned)strlen(pwd[i]);
		uint8_t *msg = NULL;
		size_t len;
		preProcessing((uint8_t*)pwd[i], str_len, &msg, &len);
		if(len != 56){
			printf ("Length of - %s - after preprocessing is not 56. Exiting\n", pwd[i]);
			exit(0);
		}
		memcpy(plaintext[i].word, msg, 64);
		++i;
		free(msg);
		msg = NULL;
	}
}

void writeToFile(word16 *md5Hash, unsigned int num, const char *filename){
	FILE *outfile = NULL;
	char *outFileName = NULL;

	char *outDir = "md5text";
	char *appendStr = "-md5";
	char *inFileName = (char*)filename;
	char *token = strsep(&inFileName, "/");
	token = strsep(&inFileName, "/");
	outFileName = (char *)malloc (strlen(outDir) + strlen(token) + strlen(appendStr) + 2);
	if(outFileName){
		outFileName[0] = '\0';
		strcat(outFileName,outDir);
		strcat(outFileName,"/");
		strcat(outFileName,token);
		strcat(outFileName,appendStr);
	}

	if ((outfile = fopen (outFileName, "w")) == NULL){
		printf ("%s can't be opened for writing\n", outFileName);
		exit(0);
	}

	unsigned int i=0;
	uint8_t result[16];
	while(i<num){
		memcpy(result,md5Hash[i].word, 16);
		unsigned int j=0;
		while(j<16){
			fprintf(outfile,"%02x", result[j]);
			//printf("%02x", result[j]);
			++j;
		}
		fprintf(outfile,"\n");
		++i;
	}
}

void printpwd(char **pwd, unsigned int num){
	unsigned int i=0;
	while(i<num) {
		char *str = pwd[i];
		printf("Pwd as Stored: %s\n",str);
		++i;
	}
	printf("Num of lines : %d\n",num);
}

void getMD5(word64* plaintext, word16 *md5Hash, unsigned int num){
	
	cudaError_t cuda_ret;
	
	Timer deviceallocation, copytodevicetime, hashtime, resultcopytime;
	
	startTime(&deviceallocation);
	word64 *ptext_d;
	word16 *md5Hash_d;
	cuda_ret = cudaMalloc((void**)&ptext_d, num*sizeof(word64));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory: %s\n", cudaGetErrorString(cuda_ret));
	cuda_ret = cudaMalloc((void**)&md5Hash_d, num*sizeof(word16));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory: %s\n", cudaGetErrorString(cuda_ret));
	
	stopTime(&deviceallocation);
    printf("Device allocation time: %f s\n", elapsedTime(deviceallocation));
    
    startTime(&copytodevicetime);
	cuda_ret = cudaMemcpy(ptext_d, plaintext, num*sizeof(word64), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device");
	stopTime(&copytodevicetime);
    printf("Copying dictionary to device: %f s\n", elapsedTime(copytodevicetime));
	
	unsigned int blocks = (unsigned int)((num - 1)/(BLOCK_SIZE*WORDS_PER_THREAD)) + 1;
	const unsigned int TOTAL_BLOCKS = (blocks > MAX_GRID_SIZE) ? MAX_GRID_SIZE : blocks;
	
	dim3 dimBlock(BLOCK_SIZE,1,1);
	dim3 dimGrid(TOTAL_BLOCKS,1,1);
	
	startTime(&hashtime);
	MD5Kernel<<<dimGrid,dimBlock>>>(ptext_d, md5Hash_d, num);
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Kernel Synchronize Error: %s\n", cudaGetErrorString(cuda_ret));
	stopTime(&hashtime);
    printf("Hash generation time: %f s\n", elapsedTime(hashtime));
	
	startTime(&resultcopytime);
	cuda_ret = cudaMemcpy(md5Hash, md5Hash_d, num*sizeof(word16), cudaMemcpyDeviceToHost);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to host");
	stopTime(&resultcopytime);
    printf("Result copying time: %f s\n", elapsedTime(resultcopytime));
}

int main(int argc, char **argv) {

    if (argc < 2) {
        printf("usage: %s 'file'\n", argv[0]);
        return 1;
    }

    const char *filename = argv[1];

    FILE *infile;
    if ((infile = fopen (filename, "r")) == NULL){
		printf ("%s can't be opened\n", filename);
		exit(0);
	}

	Timer totaltimer,filereadtimer, hostalloctime, preprocesstime, filewritetime ;
    startTime(&totaltimer);

    startTime(&filereadtimer);    
    unsigned int numPwd;
    char **pwd = NULL;
    readPwdFromFile(infile, &pwd, &numPwd);
    printf("Num Lines: %d\n",numPwd);
    //printpwd(pwd, numPwd);    
    stopTime(&filereadtimer);
    printf("File read time: %f s\n", elapsedTime(filereadtimer));

    //using static so they get allocated on heap and not on stack which can overflow
    word64 *plaintext;
    word16 *md5Hash;

    startTime(&hostalloctime);
    //plaintext = (word64*)malloc(numPwd*sizeof(word64));
    //md5Hash = (word16*)malloc(numPwd*sizeof(word16));
    
   	cudaMallocHost((void**)&plaintext, numPwd*sizeof(word64));
    cudaMallocHost((void**)&md5Hash, numPwd*sizeof(word16));
    stopTime(&hostalloctime);
    printf("Host Allocation time: %f s\n", elapsedTime(hostalloctime));
    
    startTime(&preprocesstime);    
    preProcessPwd(pwd, plaintext, numPwd);
    stopTime(&preprocesstime);
    printf("PreProcessing time: %f s\n", elapsedTime(preprocesstime));

    getMD5(plaintext, md5Hash, numPwd);
    
    startTime(&filewritetime);
    writeToFile(md5Hash, numPwd, filename);
    stopTime(&filewritetime);
    printf("File write time: %f s\n", elapsedTime(filewritetime));
    
    stopTime(&totaltimer);
    printf("Total Time: %f s\n", elapsedTime(totaltimer));

	if(infile != NULL)	fclose (infile);

    return 0;
}

