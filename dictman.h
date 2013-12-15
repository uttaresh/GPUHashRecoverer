#ifndef __DICT_MANI
#define __DICT_MANI

#include<stdlib.h>
#include<stdio.h>
#include<string.h>


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
    subs[(int)'a'] = '@';
    subs[(int)'b'] = '8';
    subs[(int)'c'] = '(';
    subs[(int)'d'] = ')';
    subs[(int)'e'] = '3';
    subs[(int)'g'] = '6';
    subs[(int)'h'] = '#';
    subs[(int)'i'] = '!';
    subs[(int)'l'] = '1';
    subs[(int)'o'] = '0';
    subs[(int)'r'] = '2';
    subs[(int)'s'] = '$';
    subs[(int)'t'] = '7';
    subs[(int)'y'] = '%';
    subs[(int)'z'] = '2';

    // Assign symbols to capital letters
    for (i='A';i<='Z';i++)
        subs[(int)i] = i + ('a' - 'A');
}

#endif

