#include "header.h"

int main()
{
    #if 0
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    EMBEDDING* model = createModel();
    train(model, s, 3, 300, 5, 10, 1, true);
    saveModel(model, false);
    free(s);
    #endif

    EMBEDDING* model = loadModelEmbeddings("model-embeddings.csv");
    //destroyModel(model);
    return 0;   
}