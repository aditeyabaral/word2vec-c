#include "header.h"

int main()
{
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    EMBEDDING* model = train(s, 2, 100, 0.1, 250, 0);
    return 0;
}


