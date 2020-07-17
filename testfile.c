#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
int getVocabularySize(char* corpus)
{
    int len = strlen(corpus);
    char* temp1 = (char*)malloc(sizeof(char)*len);
    char* temp2 = (char*)malloc(sizeof(char)*len);
    char* temp3 = (char*)malloc(sizeof(char)*len);
    strcpy(temp1, corpus);
    strcpy(temp2, "");
    strcpy(temp3, "");
    char* token1;
    char* token2;
    char* save1;
    char* save2;
    int V = 0;
    
    token1 = strtok_r(temp1, " ", &save1);
    while(token1 != NULL)
    {
        strcpy(temp3, "");
        strcpy(temp3, temp2);        
        token2 = strtok_r(temp3, " ", &save2);
        bool flag = false;
        while(token2!=NULL)
        {
            if (!strcmp(token1, token2))
            flag = true;
            token2 = strtok_r(NULL, " ", &save2);
        }
        if (!flag)
        {
            strcat(temp2, token1);
            strcat(temp2, " ");
            V++;
        }
        token1 = strtok_r(NULL, " ", &save1);
    }
    //printf("Vocab: %s\n", temp2);
    return V;
}

int main()
{
    char *s = (char*)malloc(sizeof(char)*1000);
    scanf("%[^\\0]s", s);
    printf("Input: %s\n", s);
    int V = getVocabularySize(s);
    printf("Size: %d\n", V);
    return 0;
}