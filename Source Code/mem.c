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
    //Freeing corpi
    free(model->corpus);
    free(model->clean_corpus);
    free(model->vocab);

    //Freeing weights and biases
    free2D(model->W1, model->dimension, model->vocab_size);
    free2D(model->W2, model->vocab_size, model->dimension);
    free2D(model->b1, model->dimension, 1);
    free2D(model->b2, model->vocab_size, 1);
    free2D(model->A1, model->dimension, model->batch_size);
    free2D(model->A2, model->vocab_size, model->batch_size);

    //Freeing input
    free2D(model->X, model->vocab_size, model->batch_size);
    free2D(model->Y, model->vocab_size, model->batch_size);

    //Freeing hashtable
    for(int i=0; i<model->vocab_size; ++i)
    {
        free2D(model->hashtable[i]->wordvector, 1, model->dimension);
        free2D_int(model->hashtable[i]->onehotvector, 1, model->vocab_size);
        free(model->hashtable[i]->word);
        free(model->hashtable[i]);
    }
    free(model->hashtable);

    //Free model object
    free(model);
} 