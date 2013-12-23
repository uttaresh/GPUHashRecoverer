//#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>


const unsigned int BLOCK_SIZE = 128;
const unsigned int DICT_WORDS_PER_THREAD = 10;
const unsigned int BRUTE_WORDS_NUM_BLOCKS = 10240;
//const char *characters = "abcdefghijklmnopqrstuvwxyz";
const unsigned int NUM_LETTERS = 26;
const unsigned int MAX_BRUTE_FORCE_LENGTH = 5;
const unsigned int MAX_GRID_SIZE = 65535;

FILE *logfile;

typedef struct password_t{
    char word[56];
    size_t length;
} password;

typedef struct word16_t{
    char word[16];
} word16;

