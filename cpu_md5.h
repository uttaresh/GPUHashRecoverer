#ifndef __CPU_MD5__
#define __CPU_MD5__

#define _GNU_SOURCE
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <limits.h>

/*
typedef struct word64_t{
    char word[64];
} word64;
*/

typedef struct word16_t{
    char word[16];
} word16;

typedef struct password_t{
    char word[56];
    size_t length;
} password;

void md5(password *, uint8_t *digest);
int getMD5(password *pwd, word16 *md5Hash, unsigned int num);

#endif
