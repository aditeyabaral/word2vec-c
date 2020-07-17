#include "header.h"

int main()
{
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    printf("Input: %s\n", s);
    train(s, -1, -1, 0.1, 0);
    return 0;
}


