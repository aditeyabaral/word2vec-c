#include "header.h"

int main()
{
    bool train = false;
    #if train
    char *s = (char*)malloc(sizeof(char)*INT_MAX);
    scanf("%[^\\0]s", s);
    EMBEDDING* model = createModel();
    train(model, s, 3, 300, 5, 10, 1, true);
    saveModel(model, false);
    destroyModel(model);
    free(s);
    #endif

    #if !train
    //EMBEDDING* model = loadModelEmbeddings("model-embeddings.csv");
    EMBEDDING* model = loadModelForTraining("model-embeddings.csv", NULL, NULL, NULL, NULL, NULL, NULL);
    destroyModel(model);
    #endif
    return 0;  
}