#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
struct node
{
    char *word;
    float *vector;
    bool mark;
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
    NODE **hashtable;
};
typedef struct embedding EMBEDDING;

float relu(float);
void init(EMBEDDING*);
float** createArray(int, int, int);
void display(float**, int, int);
char* remove_punctuations(char*);
int getVocabularySize(char*);
char* trim(char*);
void OneHotEncoding(EMBEDDING*, char*);
void train(EMBEDDING*, int, int, float, char*);