#include "dictman.h"
/* Function declarations */
unsigned long  mutateAndCheck(char *old_dict, unsigned int num_words, char *new_dict);
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
void mutateAndCheck(char **new_dict, char **old_dict, unsigned int num_words){

    unsigned int i, z, len, found=0;
  
    /* Loop until all mutations are done. Last mutation will break out */
    for (int z=0;z<302;z++){
        /* Copy over original dictionary */
        for (i=0;i<num_words;i++){
            memcpy(new_dict[i],old_dict[i],50*sizeof(char));
        }
       
        /* Go through each word in the dictionary */
        for (i=0;i<num_words;i++){

           len = strlen(newdict[i]);
            
           if (z==0){ 
                /* First letter uppercase */
                if (newdict[i][0] >= 'a' && newdict[i][0] <= 'z') newdict[i][0]+='A'-'a';
           }

           if (z==1){
                /* Last letter uppercase */
                if (newdict[i][len-1] >= 'a' && newdict[i][len-1] <= 'z') 
                    newdict[i][len-1] += 'A'-'a';
           }

           if (z>=2 && z<=11){
                /* Add one digit to end 
                 * iterator: z-2    */
                newdict[i][len] = '0' + z-2;
                newdict[i][len+1] = '\0';
           }

            /* Add sequence of numbers at end; e.g. 1234, 84, 1999 */
           if (z>=12 && z<=111){
                // 0 to 99
                // iterator: z-12 
                newdict[i][len] = ((z-12)/10)%10;
                newdict[i][len+1] = (z-12)%10;
                newdict[i][len+2] = '\0';
           }

           if (z>=112 && z<=231){
                // 1900 to 2020
                // iterator: z + (1900-112)
                newdict[i][len] = ((z+1900-112)/1000)%10;
                newdict[i][len+1] = ((z+1900-112)/100)%10;
                newdict[i][len+2] = ((z+1900-112)/10)%10;
                newdict[i][len+3] = (z+1900-112)%10;
                newdict[i][len+4] = '\0';
           }

            if (z>=232 && z<=251){
                // Other common sequences
                // iterator: z-232
                strcat(newdict[i],itoa(sequences[z-232]));
            }

            if (z>=252){
                /* Try all with one char-to-symbol substitution
                 * iterator: z-252
                    Ex: shitshell --> $hitshell, sh!tshell, shitsh3ll, etc. */
                    if (subs[newdict[i][z-252]]){
                        newdict[z-252] = subs[newdict[z-252]];
                    }
        
            /* Terminate loop when number 50 is substituted 
             * Right now, this is when z=252+50 = 302*/
            
            }            

            /* TODO Call Gaurav's MD5 padding, generation, and checking here */
        
        }

        
    
    }

    // TODO
    if (!found)
        printf("\nSorry. Could not crack password.");

}

/*  
    Function to append a word to the end of a dictionary.
    dict_end is the pointer to the null terminator at the end of
    the dictionary.
*/
void add2dict(char *dict_end,  char *word){




}

