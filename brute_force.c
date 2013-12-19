#include "brute_force.h"

unsigned int brute_force(){

    printf("\nStarting with brute force.\n");
    unsigned int i = 0,found=0, j=0;
    word16 md5Hash;
    password pwd;
    char *characters = "abcdefghijklmnopqrstuvwxyz";
    uint64_t total_words = 1;

	while(i < MAX_LETTERS && !found){
        total_words *= NUM_LETTERS;

        printf("Calculating with a maximum length of %d\n", i+1);
		printf("Total number of words: %lu\n\n", total_words);

		uint64_t curr_word = 0;
		uint64_t k;

		for(k=0; k<total_words && !found; k++) {
			curr_word = k;
			memset(&pwd,0,sizeof(password));
			pwd.length = i+1;
			for(j=0; j<i+1; j++) {
				pwd.word[i-j] = characters[(curr_word % NUM_LETTERS)];
				curr_word /= NUM_LETTERS;
			}
			//pwd.word[i+1] = '\0';
			//printf("%s, %d\n",pwd.word,(unsigned int)pwd.length);
            found = getMD5(&pwd, &md5Hash, 1);
		}
		++i;
	}
	return found;

}
