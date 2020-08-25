#include "header.h"

int main(int argc, char* argv[])
{
    int train_model = atoi(argv[1]);

    if (train_model == 0)
    {
        char *s = (char*)malloc(sizeof(char)*INT_MAX);
        scanf("%[^\\0]s", s);
        EMBEDDING* model = createModel();
        train(model, s, 3, 300, 0.1, 2, 0, false);
        saveModel(model, false);
        destroyModel(model);
        free(s);
    }

    else if (train_model == 1)
    {
        EMBEDDING* model = loadModelEmbeddings("model-embeddings.csv");
        destroyModel(model);
    }
    else
    {
        EMBEDDING* model = loadModelForTraining("model-embeddings.csv", "model-X.csv", "model-y.csv", "model-weights-w1.csv", "model-weights-w2.csv", "model-bias-b1.csv", "model-bias-b2.csv");
        destroyModel(model);
    }
    return 0;  
}