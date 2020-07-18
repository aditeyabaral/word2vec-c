#include "header.h"

float relu(float x)
{
    if (x>=0.0)
        return x;
    return 0.0;
}

float** softmax(float** M, int m, int n, int axis)
{
    /* Allocating space for output 2D matrix*/
    float** softmax_out = createZerosArray(m, n);

    /* Setting all values in output to be exp(corresponding value in input M) */
    for(int i=0; i<m; ++i)
    {
        for(int j=0; j<n; ++j)
        {
            softmax_out[i][j] = exp(M[i][j]);   
        }
    }

    /* Calculating column and row sums of output matrix */
    double* colsums = (double*)malloc(n*sizeof(double));
    double* rowsums = (double*)malloc(m*sizeof(double));
    memset(rowsums, 0, m*sizeof(double));
    memset(colsums, 0, n*sizeof(double));
    double sum = 0.0;
    for(int i=0; i<m; ++i)
    {
        for(int j=0; j<n; ++j)
        {
            rowsums[i] += softmax_out[i][j];
            colsums[j] += softmax_out[i][j];
            sum += softmax_out[i][j];
        }
    }
    if(axis==1)  /*Divide each element by its column sum*/
    {
        for(int i=0; i<m; ++i)
        {
            for(int j=0; j<n; ++j)
            {
                softmax_out[i][j] /= colsums[j];
            }
        }
    }
    else if(axis==0) /*Divide each element by its row sum*/
    {
        for(int i=0; i<m; ++i)
        {
            for(int j=0; j<n; ++j)
            {
                softmax_out[i][j] /= rowsums[i];
            }
        }
    }
    else /* Divide each element by matrix sum */
    {
        for(int i=0; i<m; ++i)
        {
            for(int j=0; j<n; ++j)
            {
                softmax_out[i][j] /= sum;
            }
        }
    }
    free(colsums);
    free(rowsums);
    return softmax_out;
}

float** createArray(int m, int n, int random_state)
{
    srand(random_state);
    float** array = (float**)malloc(sizeof(float*)*m);
    for(int i = 0; i<m ;i++)
    {
        array[i] = (float*)malloc(sizeof(float)*n);
        for(int j = 0; j<n; j++)
            array[i][j] = (float)rand()/(float)(RAND_MAX);
    }
    return array;
}

float** createZerosArray(int m, int n)
{
    float** array = (float**)malloc(sizeof(float*)*m);
    for(int i = 0; i<m ;i++)
    {
        array[i] = (float*)malloc(sizeof(float)*n);
        for(int j = 0; j<n; j++)
            array[i][j] = 0;
    }
    return array;
}

float** createOnesArray(int m, int n)
{
    float** array = (float**)malloc(sizeof(float*)*m);
    for(int i = 0; i<m ;i++)
    {
        array[i] = (float*)malloc(sizeof(float)*n);
        for(int j = 0; j<n; j++)
            array[i][j] = 1;
    }
    return array;
}

float** multiply(float **M1, float **M2, int m1, int n1, int m2, int n2)
{
    float **result = createZerosArray(m1, n2);
    displayArray(result, m1, n2);
    for(int i = 0; i<m1 ;i++)
    {
        for(int j=0; j<n2 ;j++)
        {
            for(int k=0; k<m2; k++)
                result[i][j]+= M1[i][k] + M2[k][j];
        }
    }
    return result;
}

float **transpose(float**A, int m, int n)
{
    float **trans = createZerosArray(n, m);
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < m; j++) 
            trans[i][j] = A[j][i];
    return trans;
}

float** createOneHot(NODE* node, EMBEDDING* model)
{
    int index = getHashvalue(node->word, model->vocab_size);
    while(model->hashtable[index] != NULL)
    {
        if(!strcmp(model->hashtable[index]->word, node->word))
            break;
        index = (index+1)%model->vocab_size;
    }
    float** oneHotVector = createZerosArray(1, model->vocab_size);
    if(oneHotVector == NULL)
    {
        printf("Failed to allocate memory for one hot vector!\n");
        return NULL;
    }
    oneHotVector[0][index] = 1;
    return oneHotVector;
}

float** getX(EMBEDDING* model, int m, char* s)
{
    float** X = createZerosArray(model->vocab_size, m);
    char *token1, *save1, *token2, *save2;
    token1 = strtok_r(s, "\n", &save1);
    int col = 0;
    while(token1 != NULL && col <= m)
    {
        float** example = createZerosArray(1, model->vocab_size);
        int ctr = 0;
        token2 = strtok_r(token1, " ", &save2);
        while(token2 != NULL && ctr <= model->context)
        {
            int index = getHashvalue(token2, model->vocab_size);
            while(model->hashtable[index] != NULL)
            {
                if(!strcmp(model->hashtable[index]->word, token2))
                    break;
                index = (index+1)%model->vocab_size;
            }
            float** oneHotVector = model->hashtable[index]->onehotvector;
            for(int j=0; j<model->vocab_size; j++)
                example[0][j]+= oneHotVector[0][j];
            token2 = strtok_r(NULL, " ", &save2);
            ctr++;
        }
        for(int j=0; j<model->vocab_size; j++)
            example[0][j]/= 2*model->context;
        for(int j = 0; j<model->vocab_size; j++)
            X[j][col] = example[0][j];
        col++;
        token1 = strtok_r(NULL, "\n", &save1);
    }
    return X;
}

float** getY(EMBEDDING* model, int m, char* s)
{
    float** y = createZerosArray(model->vocab_size, m);
    char *token, *save; 
    token = strtok_r(s, "\n", &save);
    int col = 0;
    while(token != NULL && col <= m)
    {
        int index = getHashvalue(token, model->vocab_size);
        while(model->hashtable[index] != NULL)
        {
            if(!strcmp(model->hashtable[index]->word, token))
                break;
            index = (index+1)%model->vocab_size;
        }
        float** oneHotVector = model->hashtable[index]->onehotvector;
        for(int j = 0; j<model->vocab_size; j++)
        {
            y[j][col] = oneHotVector[0][j];
        }
        col++;
        token = strtok_r(NULL, "\n", &save);
    }
    return y;
}