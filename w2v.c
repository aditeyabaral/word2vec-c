#include "header.h"

int main()
{
    //EMBEDDING* model;
    //init(model);
    int m = 10, n = 1;
    //float **a = createArray(m, n, rand());
    //display(a, m, n);
    char *s = (char*)malloc(sizeof(char)*1000);
    scanf("%s", s);
    printf("%s", s);
    char* trimmed = trim(s);
    printf("%s", trimmed);
    return 0;
}


