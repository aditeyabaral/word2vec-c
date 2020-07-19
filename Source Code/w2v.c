#include "header.h"

int main()
{
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    EMBEDDING* model = createModel();
    train(model, s, 2, 100, 0.2, 300, 0, false);
    train(model, NULL, 2, 100, 0.1, 600, 0, true);
    return 0;
}