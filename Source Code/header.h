#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <limits.h>

/*
Store a word in the vocabulary,
its embedding vector and its one hot
representation 
*/
typedef struct node
{
    char* word;
    double** wordvector;
    int** onehotvector;
}NODE;

/*
Store model hyperparameters, trainable weights
and biases, corpus, cleaned corpus and the vocabulary. 
Vocabulary is stored in the form of a hash table
*/
typedef struct embedding
{
    /* HYPERPARAMETERS */
    int context;
    float alpha;
    int dimension;
    int vocab_size;
    int corpus_length;
    int batch_size;
    int epochs;

    /* WEIGHTS AND BIASES */
    double **W1;
    double **W2;
    double **b1;
    double **b2;

    /* INPUT AND OUTPUT MATRICES */
    double** X;
    double** Y;
    double** A1;
    double** A2;
    char* vocab;
    char* corpus;
    char* clean_corpus;

    /* HASH TABLE FOR VOCABULARY STORAGE */
    NODE **hashtable;
}EMBEDDING;

/*
MODEL CREATION AND INITIALIZATION FUNCTIONS
*/

/* Initialize the hyperparameters of the model
(context width, output dimension, learning rate and number of epochs) */
void initialiseModelParameters(EMBEDDING* model, int C, int N, float alpha, int epochs);
/* Allocate memory to store the model corpus and clean it*/
void initialiseModelCorpus(EMBEDDING* model, char* corpus);
/* Allocate memory for hashtable */
void initialiseModelHashtable(EMBEDDING*);
/* Create X and Y matrices from the corpus by moving the sliding window of length 2C+1 over the corpus */
void createXandY(EMBEDDING* model, int random_state);
/* Allocate memory for model object and initalize all variables */
EMBEDDING* createModel();
/* Get final word embeddings from the trained model */
void extractEmbeddings(EMBEDDING* model);
/* Write word and its embedding to a CSV file called model-embeddings.csv */
void writeEmbeddings(EMBEDDING* model);
/* Write all hyperparameter values to model-parameters.csv */
void writeParameters(EMBEDDING* model);
/* Write clean and original corpus along with the vocabulary to model-corpus.txt */
void writeCorpus(EMBEDDING* model);
/* Write weights and biases of the NN model to model-weights-wi.csv and model-biases-bi.csv where i = 1, 2 */
void writeWeightsBiases(EMBEDDING* model);
/* Call writeEmbeddings, writeParameters and writeCorpus */
void saveModel(EMBEDDING* model, bool write_all);
/* TBD */
EMBEDDING* loadModel(char* embedding_file);
/* Create vocabulary from corpus, initialize hyperparameters and start the training loop */
void train(EMBEDDING* model, char* corpus, int C, int N, float alpha, int epochs, int random_state, bool verbose);
/*Free storage held by an arbitrary 2D array of doubles of size m * n */
void free2D(double** M, int m, int n);
/*Free storage held by an arbitrary 2D array of ints of size m * n */
void free2D_int(int** M, int m, int n);
/* Free all storage allocated for model object */
void destroyModel(EMBEDDING* model);

/*
DISPLAY FUNCTIONS
*/

/* Display a 2D array 'a' having m rows and n columns  */
void displayArray(double** a, int m, int n);
/* Display the hash table that stores the vocabulary */
void displayHashtable(EMBEDDING*);
/* Display all the hyperparameter values and weights of the model */
void displayModel(EMBEDDING* model);

/*
HASH TABLE FUNCTIONS 
*/

/* Get the hash value (index) of a given word */
int getHashvalue(char* word, int vocab_size);
/* Insert a node into the hash table */
void insert(NODE* node, EMBEDDING* model);
/* Create the hash table from the corpus */
void createHashtable(EMBEDDING*, char*);

/*
MATHEMATICAL FUNCTIONS AND MATRIX UTILITIES
*/

/* Calculate ReLU of a matrix */
double** relu(double** X, int m, int n);
/* Calculate softmax function of a matrix M over a given axis */
double** softmax(double** M, int m, int n, int axis);
/* Create matrix of size m x n and populate it with random values between 0 and 1 */
double** createArray(int m, int n, int random_state);
/* Create matrix of size m x n and populate it with 0 */
double** createZerosArray(int m, int n);
/* Create matrix of size m x n and populate it with 1 */
double** createOnesArray(int m, int n);
/* Multiply matrices M1 and M2 */
double** multiply(double **M1, double **M2, int m1, int n1, int m2, int n2);
/* Add matrices M1 and M2 */
double** add(double **M1, double **M2, int m, int n);
/* Subtract M2 from M1 */
double** subtract(double **M1, double **M2, int m, int n);
/* Multiply each element of matrix M with scalar C */
double** multiply_scalar(double **M, double C, int m, int n);
/* Transpose of matrix A having original dimensions m x n */
double** transpose(double**A, int m, int n);
/* Create one hot representation of a word contained in the node based on its index in the hashtable */
int** createOneHot(NODE* node, EMBEDDING* model);
/* Broadcast vector b to the dimension of matrix WX and then add the result*/
double** broadcast_and_add(double** WX, double **b, int m1, int n1, int m2, int n2);
/* Get the matrix X from the matrix of context words */
double** getX(EMBEDDING* model, int m, char* s);
/* Get matrix Y from matrix of target words */
double** getY(EMBEDDING* model, int m, char* s);
double dot(double** v1, double** v2, int n);
double norm(double** M, int m, int n);

/*
NEURAL NETWORK FUNCTIONS
*/

/* Implements the binary cross entropy cost function*/
float cost(EMBEDDING* model);
/* A single forward pass through the neural network */
void forward_propagation(EMBEDDING* model);
/* A single backward pass through the neural network (by calculating the gradients) */
void back_propagation(EMBEDDING* model);
/* A single forward pass, followed by a backward pass and the parameter update step */
void gradientDescent(EMBEDDING* model);


/*
TEXT PREPROCESSING FUNCTIONS
*/

/* Remove all punctuations from the corpus */
char* remove_punctuations(char*);
/* Remove leading and trailing spaces from the corpus */
char* trim(char*);
/* Get the number of words in the vocabulary from the corpus */
int getVocabularySize(EMBEDDING*);




/*
FUNCTIONALITIES
*/

double similarity(EMBEDDING* model, char* word1, char* word2);
double cosine_similarity(double** v1, double** v2, int N);
double distance(EMBEDDING* model, char* word1, char* word2);
double** getVector(EMBEDDING* model, char* word);
char* getWord(EMBEDDING* model, double** vector);