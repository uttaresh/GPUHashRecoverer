#include "support.cu"
#include "kernel.cu"

char *test;
uint8_t intTest[16];
password seq[20];


void init_seq(){
    //seq = (password *)malloc(20*sizeof(password));
    memcpy(seq[0].word,"123", 3); seq[0].length = 3;
    memcpy(seq[1].word,"1234", 4); seq[1].length = 4;
    memcpy(seq[2].word,"12345", 5);   seq[2].length = 5;
    memcpy(seq[3].word,"123456", 6);   seq[3].length = 6;
    memcpy(seq[4].word,"1234567", 7);   seq[4].length = 7;
    memcpy(seq[5].word,"12345678", 8);   seq[5].length = 8;
    memcpy(seq[6].word,"123456789", 9);   seq[6].length = 9;
    memcpy(seq[7].word,"1234567890", 10);   seq[7].length = 10;
    memcpy(seq[8].word,"696969", 6);   seq[8].length = 6;
    memcpy(seq[9].word,"111111", 6);   seq[9].length = 6;
    memcpy(seq[10].word,"1111", 4);   seq[10].length = 4;
    memcpy(seq[11].word,"1212", 4);   seq[11].length = 4;
    memcpy(seq[12].word,"7777", 4);   seq[12].length = 4;
    memcpy(seq[13].word,"1004", 4);   seq[13].length = 4;
    memcpy(seq[14].word,"2000", 4);   seq[14].length = 4;
    memcpy(seq[15].word,"4444", 4);   seq[15].length = 4;
    memcpy(seq[16].word,"2222", 4);   seq[16].length = 4;
    memcpy(seq[17].word,"6969", 4);   seq[17].length = 4;
    memcpy(seq[18].word,"9999", 4);   seq[18].length = 4;
    memcpy(seq[19].word, "3333", 4);   seq[19].length = 4;
}


void readPwdFromFile(FILE *infile, password **pwd, unsigned int *numLines){

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

    *pwd = (password*)malloc(numberOfLines*sizeof(password));
	if(*pwd == NULL){
        fprintf(logfile,"\nERROR: Memory allocation did not complete successfully! Exiting.");
        exit(0);
    }

	char *line = NULL;
	size_t len = 0;
	int read_len = 0;
	unsigned int i=0;
	unsigned int toReduce = 0;
	while (i<numberOfLines) {
		read_len = getline(&line, &len, infile);
		if(read_len != -1){
			if(line[read_len-1] == '\n')    read_len = read_len - 1;
			if(line[read_len-1] == '\r')    read_len = read_len - 1;
			if((read_len) > 45){
                //fprintf(logfile,"Skipping (too big) - %s\n",line);
                ++toReduce;
            } else {
                // (*pwd)[i-toReduce] = (char*)malloc( (read_len+1)*sizeof(char));
                memcpy((*pwd)[i-toReduce].word,line,read_len);
                (*pwd)[i-toReduce].length = read_len;
                //fprintf(logfile,"Pwd Read: %s, %d\n", (*pwd)[i], read_len);
	  		}
	  	} else {
            ++toReduce;
	  	}
        free(line);
        line = NULL;
		len = 0;
	  	i++;
	}
	*numLines = numberOfLines-toReduce;
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
		fprintf(logfile,"%s can't be opened for writing\n", outFileName);
		exit(0);
	}

	unsigned int i=0;
	uint8_t result[16];
	while(i<num){
		memcpy(result,md5Hash[i].word, 16);
		unsigned int j=0;
		while(j<16){
			fprintf(outfile,"%02x", result[j]);
			//fprintf(logfile,"%02x", result[j]);
			++j;
		}
		fprintf(outfile,"\n");
		++i;
	}
	if(outfile != NULL) fclose(outfile);
}

inline void printpwd(password *pwd){
    unsigned int i=0;
    char *str = pwd->word;
    while(i < pwd->length) {
        fprintf(logfile,"%c",str[i]);
		++i;
	}
}


void printall(password *pwd, unsigned int num){
	unsigned int i=0;
	while(i<num) {
		//char *str = pwd[i];
		fprintf(logfile,"Pwd: ");
        printpwd(&(pwd[i]));
		fprintf(logfile,"\n");
		++i;
	}
	fprintf(logfile,"Num of lines : %d\n",num);
}

void getMD5(password *pwd, password *matchPwd, const char *characters, unsigned int num){
	
	cudaError_t cuda_ret;
	
	Timer deviceallocation, hashtime, kerneltime, resultcopytime;
	
	startTime(&deviceallocation);
	password *pwd_d;
	password *matchPwd_d;
	bool *found_d;
	cuda_ret = cudaMalloc((void**)&pwd_d, num*sizeof(password));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory: %s\n", cudaGetErrorString(cuda_ret));
	cuda_ret = cudaMalloc((void**)&matchPwd_d, sizeof(password));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory: %s\n", cudaGetErrorString(cuda_ret));
	cuda_ret = cudaMalloc((void**)&found_d, sizeof(bool));
	if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory: %s\n", cudaGetErrorString(cuda_ret));

	cuda_ret = cudaMemcpyToSymbol(intTest_d, intTest , 16*sizeof(uint8_t));
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device's constant memory");
	cuda_ret = cudaMemcpyToSymbol(seq_d, seq , 20*sizeof(password));
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device's constant memory");
	cuda_ret = cudaMemcpyToSymbol(chars_d, characters, sizeof(char)*(NUM_LETTERS));
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device's constant memory");
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Device Allocation Synchronize Error: %s\n", cudaGetErrorString(cuda_ret));
	
	cuda_ret = cudaMemset (found_d, false, sizeof(bool));
	if(cuda_ret != cudaSuccess) FATAL("Unable to set memory in device");
	cuda_ret = cudaMemcpy(pwd_d, pwd, num*sizeof(password), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device");
	cuda_ret = cudaMemcpy(matchPwd_d, matchPwd, sizeof(password), cudaMemcpyHostToDevice);
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to device");
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Dictionary Copy Synchronize Error: %s\n", cudaGetErrorString(cuda_ret));
	stopTime(&deviceallocation);
	
    fprintf(logfile,"Device allocation and copy time: %f s\n", elapsedTime(deviceallocation));
    
  	unsigned int blocks = (unsigned int)((num - 1)/(BLOCK_SIZE*DICT_WORDS_PER_THREAD)) + 1;
	unsigned int TOTAL_BLOCKS = (blocks > MAX_GRID_SIZE) ? MAX_GRID_SIZE : blocks;
	
	dim3 dimBlock(BLOCK_SIZE,1,1);
	dim3 dimGrid(TOTAL_BLOCKS,1,1);
	
	fprintf(logfile,"Block Size: %d\n",BLOCK_SIZE);
	fprintf(logfile,"Total No of Blocks: %d\n",TOTAL_BLOCKS);
	
	printf("\nStarting with dictionary manipulation.\n");
	startTime(&hashtime);
	startTime(&kerneltime);
	DictKernel<<<dimGrid,dimBlock>>>(pwd_d, matchPwd_d, found_d, num);
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Kernel Synchronize Error: %s\n", cudaGetErrorString(cuda_ret));
	stopTime(&kerneltime);
    fprintf(logfile,"\nDictionary Manipulation time: %f s\n", elapsedTime(kerneltime));
	
	startTime(&resultcopytime);
	cuda_ret = cudaMemcpy(matchPwd, matchPwd_d, sizeof(password), cudaMemcpyDeviceToHost);
	cuda_ret = cudaDeviceSynchronize();
	if(cuda_ret != cudaSuccess) FATAL("Unable to copy to host");	
	stopTime(&resultcopytime);
    
	if(matchPwd[0].length < 1){
		fprintf(logfile,"\nDidn't find the password through dictionary manipulation.\n\nStarting BRUTE FORCE with max password length as %d\n\n",MAX_BRUTE_FORCE_LENGTH);
		fprintf(logfile,"Number of Threads in a block: %d\n", BLOCK_SIZE);
		if(MAX_BRUTE_FORCE_LENGTH <= 0){
			fprintf(logfile,"\nInvalid maximum length of brute force\n");
		} else {
			unsigned int i=0;
			startTime(&kerneltime);
			uint64_t total_words = 1;
			while(i < MAX_BRUTE_FORCE_LENGTH) {
				total_words *= NUM_LETTERS;
				
				fprintf(logfile,"Maximum length: %d\n", i+1);
				//fprintf(logfile,"Total number of words: %lu\n", total_words);
			
				uint64_t words_per_thread = (uint64_t)((total_words-1) / (BLOCK_SIZE*BRUTE_WORDS_NUM_BLOCKS)) + 1;
				
				unsigned int number_of_blocks;
				if(total_words / BLOCK_SIZE > BRUTE_WORDS_NUM_BLOCKS) {
					number_of_blocks = BRUTE_WORDS_NUM_BLOCKS;
				} else {
					number_of_blocks = (total_words-1)/BLOCK_SIZE + 1;
				}
				
				//fprintf(logfile,"Number of blocks: %d\n", number_of_blocks);				
				//fprintf(logfile,"Number of words per thread: %lu\n\n", words_per_thread);
				
				dim3 dimBlock(BLOCK_SIZE,1,1);
				dim3 dimGrid(number_of_blocks,1,1);
				cuda_ret = cudaMemset (found_d, false, sizeof(bool));
				if(cuda_ret != cudaSuccess) FATAL("Unable to set memory in device");
				
				BruteKernel<<<dimGrid, dimBlock>>>(matchPwd_d, i+1, words_per_thread, total_words, found_d);
				cuda_ret = cudaDeviceSynchronize();
				if(cuda_ret != cudaSuccess) FATAL("Kernel Synchronize Error: %s\n", cudaGetErrorString(cuda_ret));
	
				startTime(&resultcopytime);
				cuda_ret = cudaMemcpy(matchPwd, matchPwd_d, sizeof(password), cudaMemcpyDeviceToHost);
				cuda_ret = cudaDeviceSynchronize();
				if(cuda_ret != cudaSuccess) FATAL("Unable to copy to host");	
				stopTime(&resultcopytime);
		
				if(matchPwd[0].length > 0){
					break;
				}
				++i;
			}
			stopTime(&kerneltime);
			fprintf(logfile,"\nTotal Brute Forcing time: %f s\n", elapsedTime(kerneltime));
		}
	}
	stopTime(&hashtime);
	fprintf(logfile,"Result copying time: %f s\n", elapsedTime(resultcopytime));
	fprintf(logfile,"Total Hash time: %f s\n", elapsedTime(hashtime));
	cudaFree(pwd_d);
	cudaFree(matchPwd_d);
	cudaFree(found_d);
}

void hashToUint8(char *charHash, uint8_t intHash[]){
    char tempChar[16][3];
    int j=0;
    while(j<16){
        tempChar[j][0] = charHash[j*2];
        tempChar[j][1] = charHash[j*2+1];
        tempChar[j][2] = '\0';
        ++j;
    }
    j = 0;
    while(j<16){
        sscanf(tempChar[j], "%x", (unsigned int*)(&(intHash[j])));
        ++j;
    }
}

int main(int argc, char **argv) {

	cudaError_t cuda_ret;
	
	/*
    const char *outfilename = "log";
    if ((logfile = fopen (outfilename, "w")) == NULL){
		printf("%s can't be opened\n", outfilename);
		exit(0);
	}
	*/
	
	logfile = stdout;
	
    if (argc < 2) {
        fprintf(logfile,"usage: %s 'string hash'\n", argv[0]);
        return 1;
    }
    
  	cudaFree(0);
	init_seq();
	
	test = argv[1];
    if(strlen(test) != 32){
        fprintf(logfile,"Invalid hash. Exiting.\n");
		exit(0);
    }

    hashToUint8(test,intTest);
    //const char *filename = "plaintext/mostcommon-10k";
	const char *filename = "plaintext/sorted_dict";
    FILE *infile = NULL;
    if ((infile = fopen (filename, "r")) == NULL){
		fprintf(logfile,"%s can't be opened\n", filename);
		exit(0);
	}

	Timer totaltimer,filereadtimer, totalGPUTime ;
	Timer hostalloctime;
	//Timer filewritetime;
    startTime(&totaltimer);

    startTime(&filereadtimer);
    unsigned int numPwd;
    password *pwd = NULL;
    readPwdFromFile(infile, &pwd, &numPwd);
    fprintf(logfile,"Total Dictionary Words: %d\n",numPwd);
    //printall(pwd, numPwd);    
    stopTime(&filereadtimer);
    fprintf(logfile,"File read time: %f s\n", elapsedTime(filereadtimer));

    startTime(&hostalloctime);
   	char *characters = "abcdefghijklmnopqrstuvwxyz";
    password *matchPwd = NULL;
    cuda_ret = cudaMallocHost((void**)&matchPwd, sizeof(password));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set pinned memory on host");
    memset(matchPwd[0].word,0,64);
    matchPwd[0].length = 0;
    stopTime(&hostalloctime);
    fprintf(logfile,"Host Allocation and Mem Copy time: %f s\n", elapsedTime(hostalloctime));
    
    startTime(&totalGPUTime);
    getMD5(pwd, matchPwd, characters, numPwd);
    stopTime(&totalGPUTime);
    fprintf(logfile,"Total GPU time: %f s\n", elapsedTime(totalGPUTime));
    
	if(matchPwd[0].length > 0){
		fprintf(logfile,"\n!!!!PASSWORD FOUND!!!!\nPassword is: ");
        printpwd(matchPwd);
        fprintf(logfile,"\n\n");
	} else {
		fprintf(logfile,"Password could not be found.\n");
	}

    //startTime(&filewritetime);
    //writeToFile(md5Hash1, numPwd, filename);
    //stopTime(&filewritetime);
    //fprintf(logfile,"File write time: %f s\n", elapsedTime(filewritetime));
    
    stopTime(&totaltimer);
    fprintf(logfile,"Total Time: %f s\n", elapsedTime(totaltimer));

	free(pwd);
	cudaFree(matchPwd);
	if(infile != NULL){
		fclose (infile);
	}
	//if(logfile != NULL)	fclose (logfile);

    return 0;
}

