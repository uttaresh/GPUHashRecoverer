#include "dictman.h"
/* Function declarations */
unsigned long  mutateAndCheck(char *old_dict, unsigned int old_dict->num_entries, char *new_dict);
void add2dict(char *dict_end,  char *word);


/**
 *  CPU Function that takes in a simple dictionary of plaintext 
 *  and returns a better dictionary for stronger password cracking.
 *  
 *  @params 
 *  orig_words      pointer to original array of words
 *  orig_num_words  number of words in the original array
 *  new_dict        pointer to the new dictionary. must be freed
 *                  at end of program
**/
void mutateAndCheck(dict_t *new_dict, dict_t *orig_words,
            int orig_num_words){

    int d;
    
    // For an original string of num_entries n including null terminator, we need:
    // TODO 64 * (n^2 + 234n)
    new_dict->num_entries = orig_num_words*orig_num_words+234*orig_num_words;
    new_dict->values = (char *)malloc(64 * new_dict->num_entries);
    new_dict->values[0] = 0;
    char *dict_end = new_dict->values;

    int i=0;
    /* Iterate over entire section */
    while (i<old_dict->num_entries){
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
}

/*  
    Function to append a word to the end of a dictionary.
    dict_end is the pointer to the null terminator at the end of
    the dictionary.
*/
void add2dict(char *dict_end,  char *word){




}

int main(int argc, char **argv){
    int i;

    init_subs();
    
    dict_t orig_dict;
    char orig_values[11] = "hello";
    orig_dict.values = &orig_values;
    char *dict_end = &(orig_dict[5]);
    char new_word[5] = "owie";
    char *new_dict = 0;
    unsigned int old_dict->num_entries = 6;
    add2dict(dict_end, new_word);
    unsigned long new_dict_num_entries = mutateAndCheck(orig_dict, old_dict->num_entries, new_dict);

    for (i=0;i<new_dict_num_entries;i++){
        if (new_dict[i]) putchar(new_dict[i]);
        else putchar(' ');
    }
    
    free(new_dict);

    return 0;
}

