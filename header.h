#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
struct node
{
    char* word;
    float* wordvector;
    float** onehotvector;
};
typedef struct node NODE;

struct embedding
{
    int context;
    int alpha;
    int dimension;
    int vocab_size;
    float **W1;
    float **W2;
    float **b1;
    float **b2;
    char* vocab;
    NODE **hashtable;
};
typedef struct embedding EMBEDDING;

float relu(float);
float** softmax(float** M, int m, int n, int axis);
EMBEDDING* initialiseModelParameters(int C, int N, float alpha);
void initialiseModelHashtable(EMBEDDING*);
float** createArray(int, int, int);
float** createZerosArray(int m, int n);
float** createOnesArray(int m, int n);
void displayArray(float**, int, int);
void displayHashtable(EMBEDDING*);
char* remove_punctuations(char*);
int getVocabularySize(EMBEDDING*, char*);
char* trim(char*);
float** createOneHot(NODE* node, EMBEDDING* model);
void createHashtable(EMBEDDING*, char*);
void train(int C, int N, float alpha, char* corpus);
