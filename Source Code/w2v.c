#include "header.h"

int main()
{
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    EMBEDDING* model = createModel();
    train(model, s, 3, 300, 5, 10, 1, false);
    saveModel(model, false);
    free(s);
    destroyModel(model);
    return 0;   
}