#ifndef __DICT_MANI
#define __DICT_MANI

#include<stdlib.h>
#include<stdio.h>
#include<string.h>

typedef struct dict_t{
    char *values;
    unsigned long num_entries;
} dict_t;

// Top 20 most used number sequences
int sequences[] = { 123,1234,12345,123456,1234567,12345678,
                    123456789,1234567890,696969,111111,
                    1111,1212,7777,1004,2000,4444,2222,
                    6969,9999,3333 };

char *subs;

/* We cannot cover all substitutions. Some characters may have
 * more than one common substitution, e.g. S => $ or 5
 * To simplify our code, we currently only have one sub. per
 * character.
 */
void init_subs(){
    char i;
    // Initialize all subs to NULL initially
    subs = (char *)calloc(256,sizeof(char));    
    
    // Assign symbols to lowercase letters
    subs['a'] = '@';
    subs['b'] = '8';
    subs['c'] = '(';
    subs['d'] = ')';
    subs['e'] = '3';
    subs['g'] = '6';
    subs['h'] = '#';
    subs['i'] = '!';
    subs['l'] = '1';
    subs['o'] = '0';
    subs['r'] = '2';
    subs['s'] = '$';
    subs['t'] = '7';
    subs['y'] = '%';
    subs['z'] = '2';

    // Assign symbols to capital letters
    for (i='A';i<='Z';i++)
        subs[i] = i + ('a' - 'A');
}

#endif

