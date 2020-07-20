#include "header.h"

int main()
{
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    EMBEDDING* model = createModel();
    train(model, s, 3, 100, 0.1, 2, 0, false);
    //train(model, NULL, -1, -1, 0.01, 10000, 0, true); // to train from checkpoints
    saveModel(model, false);    // false - do not save model's corpus
    free(s);
    destroyModel(model);
    return 0;   
}