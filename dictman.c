#include "dictman.h"
/* Function declarations */
void mutateAndCheck(char **new_dict, char **old_dict, unsigned int  num_words);

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
void mutateAndCheck(char **newdict, char **olddict, unsigned int numwords){

    unsigned int i, z, len, found=0;
    char temp[10];
  
    /* Loop until all mutations are done. Last mutation will break out */
    for (z=0;z<302;z++){
        /* Copy over original dictionary */
        for (i=0;i<numwords;i++){
            memcpy(newdict[i],olddict[i],50*sizeof(char));
        }
       
        /* Go through each word in the dictionary */
        for (i=0;i<numwords;i++){

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
                sprintf(&temp,"%d",sequences[z-252]);
                strcat(newdict[i],&temp);
            }

            if (z>=252){
                /* Try all with one char-to-symbol substitution
                 * iterator: z-252
                    Ex: shitshell --> $hitshell, sh!tshell, shitsh3ll, etc. */
                    if (subs[ (int)(newdict[i][z-252]) ]){
                        newdict[i][z-252] = subs[ (int)(newdict[i][z-252]) ];
                    }
        
            
            }            

            /* Terminate loop when number 50 is substituted 
             * Right now, this is when z=302*/
            
        /* TODO Call Gaurav's MD5 padding, generation, and checking here */
        
        }

        
    
    }

    // TODO
    if (!found)
        printf("\nSorry. Could not crack password.");

}

