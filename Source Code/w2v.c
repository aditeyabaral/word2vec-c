#include "header.h"

int main()
{
    bool train_model = true;

    if (train_model)
    {
        char *s = (char*)malloc(sizeof(char)*INT_MAX);
        scanf("%[^\\0]s", s);
        EMBEDDING* model = createModel();
        train(model, s, 3, 300, 5, 10, 1, true);
        saveModel(model, false);
        destroyModel(model);
        free(s);
    }

    else
    {
        //EMBEDDING* model = loadModelEmbeddings("model-embeddings.csv");
        EMBEDDING* model = loadModelForTraining("model-embeddings.csv", NULL, NULL, NULL, NULL, NULL, NULL);
        destroyModel(model);
    }
    return 0;  
}