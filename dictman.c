#include "dictman.h"
/* Function declarations */
unsigned long  mutateAndCheck(char *old_dict, unsigned int size_old_dict, char *new_dict);
void add2dict(char *dict_end,  char *word);


/**
 *  CPU Function that takes in a simple dictionary of plaintext words separated by 
 *  a delimiter(NULL) and returns a better dictionary for stronger password cracking.
 *  @params orig_dict       pointer to array containing dictionary of words, stored
                            either in constant or shared memory. this is an array of null
                            terminated strings
            size_old_dict   size in bytes of the passed in, original dictionary
            new_dict        pointer to the new array containing the null-terminated mutated
                            words. This function will allocate memory for the array, which
                            must be freed before program termination
    
    @return                 returns the size of the new dictionary in bytes.
**/
unsigned long  mutateAndCheck(char *old_dict, 
        unsigned int size_old_dict, char *new_dict){

    int d;
    // For an original string of size n including null terminator, we need:
    // TODO n^2 + 234n + 694
    unsigned long size_new_dict = size_old_dict*size_old_dict + 234*size_old_dict + 694;
    new_dict = (char *)malloc(size_new_dict);
    new_dict[0] = 0;
    char *dict_end = new_dict;

    int i=0;
    /* Iterate over entire section */
    while (i<size_old_dict){
        int wordStart, wordEnd;
        /* Find start and end of word. Last char of word is the DELIM */
        wordStart=i;
        while(old_dict[i]!=0) i++;
        wordEnd = i;

        /* Mutate each word */ 
   
        /* Original plaintext */
        char newword[54]; // Assumes max length of original word is 54, to fit into 64-byte MD5 word
        strcpy(newword, old_dict+wordStart);

        /* First letter uppercase */
        strcpy(newword, old_dict+wordStart);
        if (newword[0] >= 'a' && newword[0] <= 'z') newword[0]+='A'-'a';
        add2dict(dict_end,newword);

        /* Last letter uppercase */
        strcpy(newword, old_dict+wordStart);
        if (newword[wordEnd-1] >= 'a' && newword[wordEnd-1] <= 'z') newword[wordEnd-1]+='A'-'a';
        add2dict(dict_end,newword);

        /* Add one digit to end */
        for (d=0;d<10;d++){
            strcpy(newword, old_dict+wordStart);
            newword[wordEnd] = '0' + d;
            newword[wordEnd+1] = '\0';
            add2dict(dict_end,newword);
        }

        /* Add sequence of numbers at end; e.g. 1234, 84, 1999 */
        // 0 to 99
        for (d=0;d<100;d++){
            strcpy(newword, old_dict+wordStart);
            newword[wordEnd] = (d/10)%10;
            newword[wordEnd+1] = d%10;
            add2dict(dict_end,newword);
        }

        // 1900 to 2020
        for (d=1900;d<2020;d++){
            strcpy(newword, old_dict+wordStart);
            newword[wordEnd] = (d/1000)%10;
            newword[wordEnd+1] = (d/100)%10;
            newword[wordEnd+2] = (d/10)%10;
            newword[wordEnd+3] = d%10;
            add2dict(dict_end,newword);
        }

        // TODO Other common sequences
        for (d=0;d<20;d++);


        /* Try all with one char-to-symbol substitution
            Ex: shitshell --> $hitshell, sh!tshell, shitsh3ll, etc. */
            for (d=wordStart;d<wordEnd;d++){
                if (subs[newword[d]]){
                    strcpy(newword, old_dict+wordStart);
                    newword[d] = subs[newword[d]];
                    add2dict(dict_end,newword);
                }
            }

        /* Increment index i by 1 to get to start of next word */
        i++;
    }
    return size_new_dict;
}

/*  
    Function to append a word to the end of a dictionary.
    dict_end is the pointer to the null terminator at the end of
    the dictionary.
*/
void add2dict(char *dict_end,  char *word){
    while(*(++dict_end) = *(word++));
}

int main(int argc, char **argv){
    int i;
    char orig_dict[11] = "hello";
    char *dict_end = &(orig_dict[5]);
    char new_word[5] = "owie";
    char *new_dict = 0;
    unsigned int size_old_dict = 6;
    add2dict(dict_end, new_word);
    unsigned long new_dict_size = mutateAndCheck(orig_dict, size_old_dict, new_dict);

    for (i=0;i<new_dict_size;i++){
        if (new_dict[i]) putchar(new_dict[i]);
        else putchar(' ');
    }

    return 0;
}

