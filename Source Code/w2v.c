#include "header.h"

int main()
{
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    EMBEDDING* model = createModel();
    train(model, s, 2, 100, 0.1, 10, 0, false);
    //train(model, NULL, -1, -1, 0.01, 10000, 0, true);
    saveModel(model, true);
    return 0;
}