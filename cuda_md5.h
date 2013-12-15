//#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

const unsigned int BLOCK_SIZE = 128;
const unsigned int WORDS_PER_THREAD = 10;
const unsigned int WORDS_IN_STREAM = 3000;
const unsigned int NUM_STREAMS = 2;

const unsigned int MAX_GRID_SIZE = 65535;
const char *DICTIONARY[] = {"plaintext/mostcommon-10k","english-395K","myspace-37K"};

typedef struct word64_t{
    char word[64];
} word64;

typedef struct word16_t{
    char word[16];
} word16;

