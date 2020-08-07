#include "header.h"

void free2D(double** M, int m, int n)
{
    if (M == NULL)
        return;

    for(int i=0;i<m;++i)
        free(M[i]);
    free(M);
}

void free2D_int(int** M, int m, int n)
{
    if (M == NULL)
        return;

    for(int i=0;i<m;++i)
        free(M[i]);
    free(M);
}

void destroyModel(EMBEDDING* model)
{
    printf("Initiating Model Deletion...\n");
    if(model == NULL)
        return;

    printf("Deleting corpi...\n");
    free(model->corpus);
    free(model->clean_corpus);
    free(model->vocab);
    

    printf("Deleting Weights & Biases...\n");
    free2D(model->W1, model->dimension, model->vocab_size);
    free2D(model->W2, model->vocab_size, model->dimension);
    free2D(model->b1, model->dimension, 1);
    free2D(model->b2, model->vocab_size, 1);
    free2D(model->A1, model->dimension, model->batch_size);
    free2D(model->A2, model->vocab_size, model->batch_size);

    printf("Deleting X & y...\n");
    free2D(model->X, model->vocab_size, model->batch_size);
    free2D(model->Y, model->vocab_size, model->batch_size);

    printf("Deleting HashTable...\n");
    for(int i=0; i<model->vocab_size; ++i)
    {
        free2D(model->hashtable[i]->wordvector, 1, model->dimension);
        free2D_int(model->hashtable[i]->onehotvector, 1, model->vocab_size);
        free(model->hashtable[i]->word);
        free(model->hashtable[i]);
    }
    free(model->hashtable);

    printf("Deleting Model...\n");
    free(model);
    printf("Model destroyed\n\n");
} 