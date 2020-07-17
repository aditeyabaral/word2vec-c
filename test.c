#include <stdio.h>
#include <stdlib.h>
#include <string.h>
char* trim(char* word)
{
    char* s = (char*)malloc(sizeof(char)*50);
    int pos1 = 0, pos2 = 0;
    while(word[pos1]==' ')
        pos1++;
    while(word[pos1]!=' ' && word[pos1]!='\0')
    {
        s[pos2] = word[pos1];
        pos1++;
        pos2++;
    }
    s[pos2] = '\0';
    return s;
}

int main()
{
    char *s = (char*)malloc(sizeof(char)*1000);
    scanf("%[^\\0]s", s);
    printf("%s\n", s);
    char* trimmed = trim(s);
    printf("%s\n", trimmed);
    return 0;
}