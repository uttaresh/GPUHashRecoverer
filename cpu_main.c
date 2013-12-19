#define _GNU_SOURCE
#include "support.h"
#include "cpu_md5.h"
#include "dictman.h"
#include "brute_force.h"

char *test;
uint8_t intTest[16];

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
        printf("\nERROR: Memory allocation did not complete successfully! Exiting.");
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

			if(read_len > 45){
                //printf("Skipping (too big) - %s\n",line);
                ++toReduce;
            } else {
                // (*pwd)[i-toReduce] = (char*)malloc( (read_len+1)*sizeof(char));
                memcpy((*pwd)[i-toReduce].word,line,read_len);
                (*pwd)[i-toReduce].length = read_len;
                //printf("Pwd Read: %s, %d\n", (*pwd)[i], read_len);
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
	//passwd = &pwd;
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

inline void printpwd(password *pwd){
    unsigned int i=0;
    char *str = pwd->word;
    while(i < pwd->length) {
        printf("%c",str[i]);
		++i;
	}
}


void printall(password *pwd, unsigned int num){
	unsigned int i=0;
	while(i<num) {
		//char *str = pwd[i];
		printf("Pwd as Stored: ");
        printpwd(&(pwd[i]));
		printf("\n");
		++i;
	}
	printf("Num of lines : %d\n",num);
}

int getMD5(password *pwd, word16 *md5Hash, unsigned int num){
	unsigned int i=0;
	//uint8_t result[16];
	while(i<num) {
		//memcpy(msg,plaintext[i].word, 64);
		md5(&(pwd[i]), (uint8_t*)md5Hash->word);

        int flag = 1;
        unsigned int j=0;
		while(j<16){
			if(intTest[j] != (uint8_t)md5Hash->word[j]){
                flag = 0;
                break;
			}
			++j;
		}
        if(flag == 1){
            printf("\n!!!!PASSWORD FOUND!!!!\nPassword is: ");
            printpwd(&(pwd[i]));
            printf("\n\n");
            return 1;
        }
		++i;
	}
	return 0;
	//free(msg);
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

    init_seq();

    if (argc < 2) {
        printf("usage: %s 'stringhash'\n", argv[0]);
        return 1;
    }

    test = argv[1];
    if(strlen(test) != 32){
        printf ("Invalid hash. Exiting.\n");
		exit(0);
    }

    hashToUint8(test,intTest);
    const char *filename = "plaintext/sorted_dict";

    FILE *infile;
    if ((infile = fopen (filename, "r")) == NULL){
		printf ("%s can't be opened\n", filename);
		exit(0);
	}

	Timer totaltimer,filereadtimer, hostalloctime, MD5time, Dicttime;
    startTime(&totaltimer);

    startTime(&filereadtimer);
    unsigned int numPwd;
    password *pwd;
    readPwdFromFile(infile, &pwd, &numPwd);
    printf("Total Dictionary Words: %d\n",numPwd);
    //printall(pwd, numPwd);
    stopTime(&filereadtimer);
    printf("File read time: %f s\n", elapsedTime(filereadtimer));

    startTime(&hostalloctime);
    //word16 *md5Hash;
    password *pwdArr;
    pwdArr = (password*)malloc(numPwd*sizeof(password));
    //md5Hash = (word16*)malloc(numPwd*sizeof(word16));
    stopTime(&hostalloctime);
    printf("Host Allocation time: %f s\n", elapsedTime(hostalloctime));

    startTime(&MD5time);
    unsigned int found = 0;
    startTime(&Dicttime);
    found = mutateAndCheck(pwdArr, pwd, numPwd);
    stopTime(&Dicttime);
    printf("Dictionary Calculation Time: %f s\n", elapsedTime(Dicttime));
    if(!found){
        printf("Couldn't find the password with dictionary manipulation\n");
        found = brute_force();
    }

    if(!found){
        printf("Sorry. Couldn't find the password\n");
    }
    stopTime(&MD5time);
    printf("MD5 Calculation time: %f s\n", elapsedTime(MD5time));

    //startTime(&filewritetime);
    //writeToFile(md5Hash, numPwd, filename);
    //stopTime(&filewritetime);
    //printf("File write time: %f s\n", elapsedTime(filewritetime));

    stopTime(&totaltimer);
    printf("Total Time: %f s\n", elapsedTime(totaltimer));

	if(infile != NULL)	fclose (infile);

    return 0;
}
