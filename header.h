#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>
struct node
{
    char* word;
    double* wordvector;
    double** onehotvector;
};
typedef struct node NODE;

struct embedding
{
    int context;
    double alpha;
    int dimension;
    int vocab_size;
    int corpus_length;
    int batch_size;
    int epochs;
    double **W1;
    double **W2;
    double **b1;
    double **b2;
    double** X;
    double** Y;
    double** Z1;
    double** Z2;
    double** A1;
    double** yhat;
    char* vocab;
    char* corpus;
    char* clean_corpus;
    NODE **hashtable;
};
typedef struct embedding EMBEDDING;

double** relu(double**, int, int);
double** softmax(double** M, int m, int n, int axis);
EMBEDDING* initialiseModelParameters(char* corpus, int C, int N, double alpha, int epochs);
void initialiseModelHashtable(EMBEDDING*);
double** createArray(int, int, int);
double** createZerosArray(int m, int n);
double** createOnesArray(int m, int n);
double** transpose(double**A, int m, int n);
double** multiply(double **M1, double **M2, int m1, int n1, int m2, int n2);
double** multiply_scalar(double **M, double C, int m, int n);
double** add(double **M1, double **M2, int m, int n);
double** subtract(double **M1, double **M2, int m, int n);
void displayArray(double**, int, int);
void displayHashtable(EMBEDDING*);
void displayModel(EMBEDDING* model);
char* remove_punctuations(char*);
int getVocabularySize(EMBEDDING*);
int getHashvalue(char*, int);
char* trim(char*);
double** createOneHot(NODE* node, EMBEDDING* model);
void createHashtable(EMBEDDING*, char*);
void createXandY(EMBEDDING* model, int random_state);
double** getX(EMBEDDING* model, int m, char* s);
double** getY(EMBEDDING* model, int m, char* s);
EMBEDDING* train(char* corpus, int C, int N, double alpha, int epochs, int random_state);
void gradientDescent(EMBEDDING* model);
void back_propagation(EMBEDDING* model);
void forward_propagation(EMBEDDING* model);
double** broadcast_and_add(double** WX, double **b, int m1, int n1, int m2, int n2);
double cost(EMBEDDING* model);
