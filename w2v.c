#include "header.h"

int main()
{
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    EMBEDDING* model = train(s, -1, -1, 0.01, 100, 0);
    return 0;
}


