#DEFINE DELIM 0
/*
   Kernel that takes in a simple dictionary of plaintext words separated by 
   a delimiter and checks if that plaintext or mutations of the 
   plaintext yield an MD5 hash match.
*/
__global__ void  mutateAndCheck(char *wordArray, char *targetHash){

    /* All words that BEGIN in our assigned range should have their hashes
       checked by our thread. Each thread is in charge of a section of
       48 chars */
    int i=48*(BlockDim.x*BlockIdx.x+ThreadIdx.x);    // Starting index
    
    // Does a word start exactly at the start of our section?
    if (i > 0) if (wordArray[i-1]==DELIM) i--;
    /* Go to start of next word */
    while(wordArray[i]!=DELIM) i++;
    i++;

    /* Iterate over entire section */
    while (i<48){
        int wordStart, wordEnd;
        /* Find start and end of word. Last char of word is the DELIM */
        wordStart=i;
        while(wordArray[i]!=DELIM) i++;
        wordEnd = i;

        /* Mutate each word and check if it matches our hash */
       
            /* Original plaintext */

            /* First letter uppercase */

            /* Last letter uppercase */

            /* Add one digit to end */

            /* Add sequence of numbers at end; e.g. 1234, 84, 1999 */

            /* Try all with one char-to-symbol substitution
                Ex: shitshell --> $hitshell, sh!tshell, shitsh3ll, etc. */

        /* Increment index i by 1 to get to start of next word */
    }
}

