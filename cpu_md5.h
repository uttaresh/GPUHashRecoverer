#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

void md5(uint8_t*, size_t, uint8_t*);
void preProcessing(uint8_t *initial_msg, size_t initial_len, uint8_t **final_msg, size_t *len);

typedef struct word64_t{
    char word[64];
} word64;

typedef struct word16_t{
    char word[16];
} word16;
