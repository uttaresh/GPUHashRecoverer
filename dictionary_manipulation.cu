/**
 *  Kernel that takes in a simple dictionary of plaintext words separated by 
 *  a delimiter(NULL) and checks if that plaintext or mutations of the 
 *  plaintext yield an MD5 hash match.
 *  @params wordArray       pointer to array containing dictionary of words, stored
                            either in constant or shared memory. this is an array of null
                            terminated strings
            targetHash      pointer to the 16 byte target hash. this is what we 
                            will generate with the correct password. stored in const mem
            sequences       pointer to array of 100 most common sequences of numbers that 
                            are included in passwords. this is an array of null terminated
                            strings
            substitutions   pointer to array of common character to symbol substitutions.
                            all even positions hold original char. all odd positions hold
                            substitutions. e.g. a -> @, B -> 8, c -> (, etc.
**/
__global__ void  mutateAndCheck(char *wordArray, char *targetHash, 
        char *sequences, char *substitutions){

    /* All words that BEGIN in our assigned range should have their hashes
       checked by our thread. Each thread is in charge of a section of
       48 chars */
    int i=48*(BlockDim.x*BlockIdx.x+ThreadIdx.x);    // Starting index
    
    // Does a word start exactly at the start of our section?
    if (i > 0) if (wordArray[i-1]==0) i--;
    /* Go to start of next word */
    while(wordArray[i]!=0) i++;
    i++;

    /* Iterate over entire section */
    while (i<48){
        int wordStart, wordEnd;
        /* Find start and end of word. Last char of word is the DELIM */
        wordStart=i;
        while(wordArray[i]!=0) i++;
        wordEnd = i;

        /* Mutate each word and check if it matches our hash */
   
        /* Original plaintext */
        char newword[25]; // Assumes max length of original word is 15, to allow for 10 digits at end
        strcpy(newword, wordArray+wordStart);
        generateMD5(newword);

        /* First letter uppercase */
        strcpy(newword, wordArray+wordStart);
        if (newword[0] >= 'a' && newword[0] <= 'z') newword[0]+='A'-'a';
        generateMD5(newword);

        /* Last letter uppercase */
        strcpy(newword, wordArray+wordStart);
        if (newword[wordEnd-1] >= 'a' && newword[wordEnd-1] <= 'z') newword[wordEnd-1]+='A'-'a';
        generateMD5(newword);

        /* Add one digit to end */
        for (int d=0;d<10;d++){
            strcpy(newword, wordArray+wordStart);
            newword[wordEnd] = '0' + d;
            newword[wordEnd+1] = '\0';
            generateMD5(newword);
        }

        /* Add sequence of numbers at end; e.g. 1234, 84, 1999 */
        // 0 to 99
        for (int d=0;d<100;d++){
            strcpy(newword, wordArray+wordStart);
            newword[wordEnd] = (d/10)%10;
            newword[wordEnd+1] = d%10;
            generateMD5(newword);
        }

        // 1900 to 2020
        for (int d=1900;d<2020;d++){
            strcpy(newword, wordArray+wordStart);
            newword[wordEnd] = (d/1000)%10;
            newword[wordEnd+1] = (d/100)%10;
            newword[wordEnd+2] = (d/10)%10;
            newword[wordEnd+3] = d%10;
            generateMD5(newword);
        }

        // 100 most common numbers of size > 2; e.g. 123, 1234, 321, 00000, etc.
        for (int d=0;d<100;d++){
            // Try number sequence before and after word
            strcpy(newword, wordArray+wordStart);
            strcat(newword, sequences+d);
            generateMD5(newword);
            strcpy(newword, sequences+d);
            strcat(newword, wordArray+wordStart);
            generateMD5(newword);
        } 

        /* Try all with one char-to-symbol substitution
            Ex: shitshell --> $hitshell, sh!tshell, shitsh3ll, etc. */
            for (int d=wordStart;d<wordEnd;d++){
                strcpy(newword, wordArray+wordStart);
                newword[d] = substitutions[newword[d]];
                generateMD5(newword);
            }

        /* Increment index i by 1 to get to start of next word */
        i++;
    }
}

