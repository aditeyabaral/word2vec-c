#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
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

void displayArray(float **a, int m, int n)
{
    for(int i = 0; i<m ;i++)
    {
        for(int j = 0; j<n; j++)
            printf("%f ", a[i][j]);
        printf("\n");
    }
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

int main()
{
    int m1 = 1, n1 = 2, m2 = 2, n2 = 5;
    float **M1 = createArray(m1, n1, 0);
    float **M2 = createArray(m2, n2, 0);
    displayArray(M1, m1, n1);
    displayArray(M2, m2, n2);
    float **result = multiply(M1, M2, m1, n1, m2, n2);
    displayArray(result, m1, n2);
}