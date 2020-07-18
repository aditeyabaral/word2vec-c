#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

void func(char* s)
{
    int len = strlen(s);
    char* temp = (char*)malloc(sizeof(char)*len);
    strcpy(temp, s);
    char *token, *save;
    token = strtok_r(temp, " ", &save);
    while(token!=NULL)
    {
        printf("Token: %s\n", token);
        token = strtok_r(NULL, " ", &save);
    }
}

int main()
{
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    func(s);
    return 0;
}