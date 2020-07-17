#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
struct node
{
    char *word;
    float *vector;
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
void initialiseModelParameters(EMBEDDING*);
void initialiseModelHashtable(EMBEDDING*);
float** createArray(int, int, int);
void displayArray(float**, int, int);
void displayHashtable(EMBEDDING*);
char* remove_punctuations(char*);
int getVocabularySize(EMBEDDING*, char*);
char* trim(char*);
void createHashtable(EMBEDDING*, char*);
void train(EMBEDDING* model, int C, int N, float alpha, char* corpus);