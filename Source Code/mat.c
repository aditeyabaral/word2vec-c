#include "header.h"

double** relu(double** X, int m, int n)
{
    double** result = createZerosArray(m, n);
    for(int i = 0; i<m; i++)
    {
        for(int j=0; j<n; j++)
        {
            if (X[i][j]<=0.0)
                result[i][j] = 0.0;
            else
                result[i][j] = X[i][j];
        }
    }
    return result;
}

double** softmax(double** M, int m, int n, int axis)
{
    /* Allocating space for output 2D matrix*/
    double** softmax_out = createZerosArray(m, n);

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
    if(axis==0)  /*Divide each element by its column sum*/
    {
        for(int i=0; i<m; ++i)
        {
            for(int j=0; j<n; ++j)
            {
                softmax_out[i][j] /= colsums[j];
            }
        }
    }
    else if(axis==1) /*Divide each element by its row sum*/
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

double** createArray(int m, int n, int random_state)
{
    srand(random_state);
    double** array = (double**)malloc(sizeof(double*)*m);
    for(int i = 0; i<m ;i++)
    {
        array[i] = (double*)malloc(sizeof(double)*n);
        for(int j = 0; j<n; j++)
            array[i][j] = ((double)rand()/(double)(RAND_MAX))/10;
    }
    return array;
}

double** createZerosArray(int m, int n)
{
    double** array = (double**)malloc(sizeof(double*)*m);
    for(int i = 0; i<m ;i++)
    {
        array[i] = (double*)malloc(sizeof(double)*n);
        for(int j = 0; j<n; j++)
            array[i][j] = 0;
    }
    return array;
}

double** createOnesArray(int m, int n)
{
    double** array = (double**)malloc(sizeof(double*)*m);
    for(int i = 0; i<m ;i++)
    {
        array[i] = (double*)malloc(sizeof(double)*n);
        for(int j = 0; j<n; j++)
            array[i][j] = 1;
    }
    return array;
}

double** multiply(double **M1, double **M2, int m1, int n1, int m2, int n2)
{
    double **result = createZerosArray(m1, n2);
    for(int i = 0; i<m1 ;i++)
    {
        for(int j=0; j<n2 ;j++)
        {
            for(int k=0; k<m2; k++)
                result[i][j]+= M1[i][k] * M2[k][j];
        }
    }
    return result;
}

double** add(double **M1, double **M2, int m, int n)
{
    double **result = createZerosArray(m, n);
    for(int i = 0; i<m; i++)
    {
        for(int j=0; j<n ;j++)
            result[i][j] = M1[i][j] + M2[i][j];
    }
    return result;
}

double** subtract(double **M1, double **M2, int m, int n)
{
    double **result = createZerosArray(m, n);
    for(int i = 0; i<m; i++)
    {
        for(int j=0; j<n ;j++)
            result[i][j] = M1[i][j] - M2[i][j];
    }
    return result;
}

double** multiply_scalar(double **M, double C, int m, int n)
{
    double **result = createZerosArray(m, n);
    for(int i = 0; i<m; i++)
    {
        for(int j=0; j<n ;j++)
            result[i][j] = M[i][j]*C;
    }
    return result;
}

double **transpose(double**A, int m, int n)
{
    double **trans = createZerosArray(n, m);
    for (int i = 0; i < n; i++) 
        for (int j = 0; j < m; j++) 
            trans[i][j] = A[j][i];
    return trans;
}

int** createOneHot(NODE* node, EMBEDDING* model)
{
    int index = getHashvalue(node->word, model->vocab_size);
    while(model->hashtable[index] != NULL)
    {
        if(!strcmp(model->hashtable[index]->word, node->word))
            break;
        index = (index+1)%model->vocab_size;
    }
    int** oneHotVector = (int**)malloc(sizeof(int*));
    oneHotVector[0] = (int*)malloc(sizeof(int)*model->vocab_size);
    for(int i=0; i < model->vocab_size; i++)
        oneHotVector[0][i] = 0;
    oneHotVector[0][index] = 1;
    return oneHotVector;
}

double** broadcast_and_add(double** WX, double **b, int m1, int n1, int m2, int n2)
{
    double **Z1 = createZerosArray(m1, n1);
    for(int i=0; i<n1; i++)
    {
        for(int j=0; j<m1; j++)
            Z1[j][i] = WX[j][i] + b[j][0];
    }
    return Z1;
}

double dot(double** v1, double** v2, int n)
{
    double result = 0;
    for(int i = 0; i<n; i++)
        result+= v1[0][i] * v2[0][i];
    return result;
}

double cosine_similarity(double** v1, double** v2, int N)
{
    return dot(v1, v2, N)/(norm(v1, 1, N)*norm(v2, 1, N));
}

double similarity(EMBEDDING* model, char* word1, char* word2)
{
    double** v1 = getVector(model, word1);
    double** v2 = getVector(model, word2);
    if(v1 == NULL)
    {
        printf("%s does not belong in vocabulary.\n", word1);
        return -1;
    }
    if(v2 == NULL)
    {
        printf("%s does not belong in vocabulary.\n", word2);
        return -1;
    }
    return cosine_similarity(v1, v2, model->dimension);
}

double distance(EMBEDDING* model, char* word1, char* word2)
{
    double** v1 = getVector(model, word1);
    double** v2 = getVector(model, word2);
    return 1.0-cosine_similarity(v1, v2, model->dimension);
}

double norm(double** M, int m, int n)
{
    double result = 0;
    for(int i = 0; i<m; i++)
    {
        for(int j = 0; j < n; j++)
            result+= M[i][j]*M[i][j];
    }
    result = sqrt(result);
    return result;
}

double** getX(EMBEDDING* model, int m, char* s)
{
    double** X = createZerosArray(model->vocab_size, m);
    char *token1, *save1, *token2, *save2;
    token1 = strtok_r(s, "\n", &save1);
    int col = 0;
    while(token1 != NULL && col <= m)
    {
        double** example = createZerosArray(1, model->vocab_size);
        int ctr = 0;
        token2 = strtok_r(token1, " ", &save2);
        while(token2 != NULL && ctr <= 2*model->context)
        {
            int index = getHashvalue(token2, model->vocab_size);
            while(model->hashtable[index] != NULL)
            {
                if(!strcmp(model->hashtable[index]->word, token2))
                    break;
                index = (index+1)%model->vocab_size;
            }
            int** oneHotVector = model->hashtable[index]->onehotvector;
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
        free2D(example, 1, model->vocab_size);
        token1 = strtok_r(NULL, "\n", &save1);
    }
    return X;
}

double** getY(EMBEDDING* model, int m, char* s)
{
    double** y = createZerosArray(model->vocab_size, m);
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
        int** oneHotVector = model->hashtable[index]->onehotvector;
        for(int j = 0; j<model->vocab_size; j++)
            y[j][col] = oneHotVector[0][j];
        col++;
        token = strtok_r(NULL, "\n", &save);
    }
    return y;
}