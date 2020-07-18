#include "header.h"

void displayArray(float **a, int m, int n)
{
    for(int i = 0; i<m ;i++)
    {
        for(int j = 0; j<n; j++)
            printf("%f ", a[i][j]);
        printf("\n");
    }
    printf("\n");
}

void displayHashtable(EMBEDDING* model)
{
    for(int i=0;i<model->vocab_size;i++)
    {
        if(model->hashtable[i] != NULL)
        {
            printf("%s\n", model->hashtable[i]->word);
            for(int j=0; j<model->vocab_size; ++j)
                printf("%.1f ", model->hashtable[i]->onehotvector[0][j]);
            printf("\n\n");
        }
    }
}

void displayModel(EMBEDDING* model)
{
    printf("Input: %s\n\n", model->corpus);
    printf("Cleaned Text : %s\n\n", model->clean_corpus);
    printf("Vocabulary: %s\n\n", model->vocab);
    printf("C: %d\nN: %d\nalpha: %f\nVocab Size: %d\nBatch Size: %d\n\n", model->context, model->dimension, model->alpha, model->vocab_size, model->batch_size);
    printf("Hashtable: \n");
    displayHashtable(model);
    printf("\nX: \n\n");
    displayArray(model->X, model->vocab_size, model->batch_size);
    printf("\ny: \n\n");
    displayArray(model->Y, model->vocab_size, model->batch_size);
    printf("\nW1: \n");
    displayArray(model->W1, model->dimension, model->vocab_size);
    printf("\nW2: \n");
    displayArray(model->W2, model->vocab_size, model->dimension);
    printf("\nb1: \n");
    displayArray(model->b1, model->dimension, 1);
    printf("\nb2: \n");
    displayArray(model->b2, model->vocab_size, 1);
}
