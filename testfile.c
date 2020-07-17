#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <limits.h>

void createXandY(char* s, int context)
{
    int num_words = 0;
    int len = strlen(s);
    char* token1 , *save1;
    char* temp1 = (char*)malloc(sizeof(char)*len);
    strcpy(temp1, s);
    token1 = strtok_r(temp1, " ", &save1);
    while(token1 != NULL)
    {
        num_words++;
        token1 = strtok_r(NULL, " ", &save1);
    }
    if ((2*context+1)>=num_words)
    {
        printf("Not enough words available for context. Window >= number of words.\n");
        return;
    }
    int ctr1 = 1, ctr2 = 1;
    char* temp2 = (char*)malloc(sizeof(char)*len);
    strcpy(temp1, s);
    token1 = strtok_r(temp1, " ", &save1);
    char* token2, *save2;

    int m = 1+(num_words%(2*context+1));
    char* X_words = (char*)malloc(sizeof(char)*INT_MAX);
    char* y_words = (char*)malloc(sizeof(char)*INT_MAX);

    while(token1 != NULL && ctr1 <= (num_words-2*context))
    {
        ctr2 = 1;
        strcpy(temp2, s);
        token2 = strtok_r(temp2, " ", &save2);
        while(token2 != NULL && ctr2!=ctr1)
        {
            token2 = strtok_r(NULL, " ", &save2);
            ctr2++;
        }
        for(int i=1;i<=context;i++)
        {
            strcat(X_words, token2);
            strcat(X_words, " ");
            token2 = strtok_r(NULL, " ", &save2);
        }
        strcat(y_words, token2);
        strcat(y_words, "\n");
        token2 = strtok_r(NULL, " ", &save2);
        for(int i=1;i<=context;i++)
        {
            strcat(X_words, token2);
            strcat(X_words, " ");
            token2 = strtok_r(NULL, " ", &save2);
        }
        strcat(X_words, "\n");
        ctr1++;
        token1 = strtok_r(NULL, " ", &save1);
    }

    printf("Input: \n");
    printf("%s\n\n", s);
    printf("X: \n");
    printf("%s\n\n", X_words);
    printf("y: \n");
    printf("%s\n\n", y_words);
}



int main()
{
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    createXandY(s, 2);
    return 0;
}