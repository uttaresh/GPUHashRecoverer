#include "dictman.h"
/* Function declarations */

void init_seq(){

    seq = (password *)malloc(20*sizeof(password));

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
unsigned int mutateAndCheck(password *newdict, password *olddict, unsigned int numwords){

    printf("Starting with dictionary manipulation.\n");
    unsigned int i, found=0;
    int z;
    word16 md5Hash;

    /* Loop until all mutations are done. Last mutation will break out */
    for (z=-1;z<252 && !found;z++){
        //printf("Iteration: %d\n",z);
        /* Copy over original dictionary */
        memcpy(newdict,olddict,numwords*sizeof(password));

        if(z != -1){
            /* Go through each word in the dictionary */
            for (i=0;i<numwords;i++){

               if (z==0){
                    /* First letter uppercase */
                    if (newdict[i].word[0] >= 'a' && newdict[i].word[0] <= 'z')
                        newdict[i].word[0] +='A'-'a';
               }

               if (z==1){
                    /* Last letter uppercase */
                    size_t len = newdict[i].length;
                    if (newdict[i].word[len-1] >= 'a' && newdict[i].word[len-1] <= 'z')
                        newdict[i].word[len-1] += 'A'-'a';
               }

               if (z>=2 && z<=11){
                    /* Add one digit to end
                     * iterator: z-2    */
                    size_t len = newdict[i].length;
                    newdict[i].word[len] = '0' + z-2;
                    newdict[i].length += 1;
               }

                /* Add sequence of numbers at end; e.g. 1234, 84, 1999 */
               if (z>=12 && z<=111){
                    // 0 to 99
                    // iterator: z-12
                    size_t len = newdict[i].length;
                    newdict[i].word[len] = '0' + ((z-12)/10)%10;
                    newdict[i].word[len+1] = '0' + (z-12)%10;
                    newdict[i].length += 2;
               }

               if (z>=112 && z<=231){
                    // 1900 to 2020
                    // iterator: z + (1900-112)
                    size_t len = newdict[i].length;
                    newdict[i].word[len] = '0' + ((z+1900-112)/1000)%10;
                    newdict[i].word[len+1] = '0' + ((z+1900-112)/100)%10;
                    newdict[i].word[len+2] = '0' + ((z+1900-112)/10)%10;
                    newdict[i].word[len+3] = '0' + (z+1900-112)%10;
                    newdict[i].length += 4;
               }

                if (z>=232 && z<=251){
                    // Other common sequences
                    // iterator: z-232
                    //sprintf(&temp,"%s",sequences[z-252]);
                    size_t len = newdict[i].length;
                    memcpy(&(newdict[i].word[len]),seq[z-232].word,seq[z-232].length);
                    newdict[i].length = len + seq[z-232].length;
                }

                /* Terminate loop when number 50 is substituted
                 * Right now, this is when z=252*/

                /* TODO Call Gaurav's MD5 padding, generation, and checking here */

            }
        }

        found = getMD5(newdict, &md5Hash, numwords);
    }

    return found;

}

