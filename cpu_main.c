#define _GNU_SOURCE
#include "support.h"
#include "cpu_md5.h"

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

    *pwd = malloc(numberOfLines * sizeof(char*));

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
                printf("Skipping (too big) - %s\n",line);
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

void getMD5(word64 *plaintext, word16 *md5Hash, unsigned int num){
	unsigned int i=0;
	uint8_t result[16];
	char *msg;
	msg = (char*)malloc(64);
	while(i<num) {
		memcpy(msg,plaintext[i].word, 64);
		md5((uint8_t*)msg, 56, result);
		memcpy(md5Hash[i].word, result, 16);
		++i;
	}
	free(msg);
}

void writeToFile(word16 *md5Hash, unsigned int num, const char *filename){
	FILE *outfile = NULL;
	char *outFileName = NULL;

	char *outDir = "md5text";
	char *appendStr = "-md5";
	char *inFileName = (char*)filename;
	char *token = strsep(&inFileName, "/");
	token = strsep(&inFileName, "/");
	outFileName = malloc (strlen(outDir) + strlen(token) + strlen(appendStr) + 2);
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

	Timer totaltimer,filereadtimer, hostalloctime, preprocesstime, MD5time, filewritetime ;
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
    plaintext = (word64*)malloc(numPwd*sizeof(word64));
    md5Hash = (word16*)malloc(numPwd*sizeof(word16));
    stopTime(&hostalloctime);
    printf("Host Allocation time: %f s\n", elapsedTime(hostalloctime));

    startTime(&preprocesstime);
    preProcessPwd(pwd, plaintext, numPwd);
    stopTime(&preprocesstime);
    printf("PreProcessing time: %f s\n", elapsedTime(preprocesstime));

    startTime(&MD5time);
    getMD5(plaintext, md5Hash, numPwd);
    stopTime(&MD5time);
    printf("MD5 generation time: %f s\n", elapsedTime(MD5time));

    startTime(&filewritetime);
    writeToFile(md5Hash, numPwd, filename);
    stopTime(&filewritetime);
    printf("File write time: %f s\n", elapsedTime(filewritetime));

    stopTime(&totaltimer);
    printf("Total Time: %f s\n", elapsedTime(totaltimer));

	if(infile != NULL)	fclose (infile);

    return 0;
}
