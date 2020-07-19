#include "header.h"

void displayArray(double **a, int m, int n)
{
    for(int i = 0; i<m ;i++)
    {
        for(int j = 0; j<n; j++)
            printf("%lf ", a[i][j]);
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
            for(int j=0; j<model->dimension; ++j)
                printf("%.1lf ", model->hashtable[i]->wordvector[0][j]);
            printf("\n\n");
        }
    }
}

void displayModel(EMBEDDING* model)
{
    printf("Input: %s\n\n", model->corpus);
    printf("Cleaned Text : %s\n\n", model->clean_corpus);
    printf("Vocabulary: %s\n\n", model->vocab);
    printf("C: %d\n", model->context);
    printf("N: %d\n", model->dimension);
    printf("alpha: %f\n", model->alpha);
    printf("Vocabulary Size: %d\n", model->vocab_size);
    printf("Batch Size: %d\n", model->batch_size);
    printf("Epochs: %d\n\n", model->epochs);
    printf("Hashtable: \n");
    displayHashtable(model);
    printf("\nX: \n\n");
    displayArray(model->X, model->vocab_size, model->batch_size);
    printf("\ny: \n\n");
    displayArray(model->Y, model->vocab_size, model->batch_size);
    printf("\nW1: \n\n");
    displayArray(model->W1, model->dimension, model->vocab_size);
    printf("\nW2: \n\n");
    displayArray(model->W2, model->vocab_size, model->dimension);
    printf("\nb1: \n\n");
    displayArray(model->b1, model->dimension, 1);
    printf("\nb2: \n\n");
    displayArray(model->b2, model->vocab_size, 1);
}
