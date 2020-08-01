#include "header.h"

bool checkFileExists(char* filename)
{
    if (filename == NULL)
        return false;
    FILE* fp = fopen(filename, "r");
    bool check = true;
    if (fp == NULL)
    {
        printf("%s not found.\n", filename);
        check = false;
    }
    fclose(fp);
    return check;
}

void getFileDimensions(char* filename, int *m, int *n)
{
    FILE* fp = fopen(filename, "r");
    char* temp = (char*)malloc(sizeof(char)*INT_MAX);
    char *line, *token, *save;
    *m = 0;
    *n = 0;
    bool flag = false;
    while(fgets(temp, INT_MAX, fp))
    {
        line = trim(temp);
        *m = *m + 1;
        token = strtok_r(line, ",", &save);
        while(token!=NULL && !flag)
        {
            *n = *n + 1;
            token = strtok_r(NULL, ",", &save);
        }
        flag = true;
        free(line);
    }
    free(temp);
    fclose(fp);
}

double** getMatrixFromFile(EMBEDDING* model, char* filename)
{
    int m, n, row, col;
    getFileDimensions(filename, &m, &n);
    double** M = createZerosArray(m, n);

    FILE* fp = fopen(filename, "r");
    char* temp = (char*)malloc(sizeof(char)*INT_MAX);
    char *line, *token, *save, *ptr;
    row = 0;
    while(fgets(temp, INT_MAX, fp))
    {
        line = trim(temp);
        token = strtok_r(line, ",", &save);
        col = 0;
        while(token!=NULL)
        {
            M[row][col] = strtod(token, &ptr);
            col++;
            token = strtok_r(NULL, ",", &save);
        }
        row++;
        free(line);
    }
    free(temp);
    fclose(fp);
    return M;
}

void writeEmbeddings(EMBEDDING* model)
{
    FILE* fp  = fopen("model-embeddings.csv", "w");
    for(int i = 0; i < model->vocab_size; i++)
    {
        fprintf(fp, "%s,", model->hashtable[i]->word);
        for(int j = 0; j < model->dimension; j++)
            fprintf(fp, "%lf,", model->hashtable[i]->wordvector[0][j]);
        fprintf(fp, "%c", '\n');
    }
    fclose(fp);
    printf("Embedding saved...\n");
}

void writeParameters(EMBEDDING* model)
{
    FILE* fp  = fopen("model-parameters.csv", "w");
    fprintf(fp, "alpha,%f,\n", model->alpha);
    fprintf(fp, "C,%d,\n", model->context);
    fprintf(fp, "N,%d,\n", model->dimension);
    fprintf(fp, "Vocabulary Size,%d,\n", model->vocab_size);
    fprintf(fp, "Batch Size,%d,\n", model->batch_size);
    fprintf(fp, "Corpus Length,%d,\n", model->corpus_length);
    fprintf(fp, "Epochs,%d,\n", model->epochs);
    fclose(fp);
    printf("Parameters saved...\n");
}

void writeCorpus(EMBEDDING* model)
{
    FILE* fp  = fopen("model-corpus.txt", "w");
    fprintf(fp, "Cleaned Corpus,%s,\n\n", model->clean_corpus);
    fprintf(fp, "Vocabulary,%s,\n\n", model->vocab);
    fprintf(fp, "Corpus,%s,\n\n", model->corpus);
    fclose(fp);
    printf("Corpus saved...\n");
}

void writeXY(EMBEDDING* model)
{
    FILE* fp  = fopen("model-X.csv", "w");
    for(int i = 0; i<model->vocab_size; i++)
    {
        for(int j = 0; j<model->batch_size; j++)
            fprintf(fp, "%lf,", model->X[i][j]);
        fprintf(fp, "%c", '\n');
    }

    fp  = fopen("model-y.csv", "w");
    for(int i = 0; i<model->vocab_size; i++)
    {
        for(int j = 0; j<model->batch_size; j++)
            fprintf(fp, "%lf,", model->Y[i][j]);
        fprintf(fp, "%c", '\n');
    }

    fclose(fp);
    printf("X and y saved...\n");
}

void writeWeightsBiases(EMBEDDING* model)
{
    FILE* fp  = fopen("model-weights-w1.csv", "w");
    for(int i = 0; i<model->dimension; i++)
    {
        for(int j = 0; j<model->vocab_size; j++)
            fprintf(fp, "%lf,", model->W1[i][j]);
        fprintf(fp, "%c", '\n');
    }

    fp  = fopen("model-weights-w2.csv", "w");
    for(int i = 0; i<model->vocab_size; i++)
    {
        for(int j = 0; j<model->dimension; j++)
            fprintf(fp, "%lf,", model->W2[i][j]);
        fprintf(fp, "%c", '\n');
    }
    printf("Weights saved...\n");

    fp  = fopen("model-bias-b1.csv", "w");
    for(int i = 0; i<model->dimension; i++)
        fprintf(fp, "%lf,", model->b1[i][0]);
    fprintf(fp, "%c", '\n');

    fp  = fopen("model-bias-b2.csv", "w");
    for(int i = 0; i<model->vocab_size; i++)
        fprintf(fp, "%lf,", model->b2[i][0]);
    fprintf(fp, "%c", '\n');

    fclose(fp);
    printf("Biases saved...\n");
}

void saveModel(EMBEDDING* model, bool write_all)
{
    writeEmbeddings(model);
    writeParameters(model);
    writeWeightsBiases(model);
    if (write_all)
        writeCorpus(model);
}


EMBEDDING* loadModelForTraining(char* embedding_filename, char* X_filename, char* Y_filename, char* W1_filename, char* W2_filename, char* b1_filename, char* b2_filename)
{
    if(!(checkFileExists(X_filename) && checkFileExists(Y_filename) && checkFileExists(W1_filename) 
        && checkFileExists(W2_filename) && checkFileExists(b1_filename) && checkFileExists(b2_filename)))
    {
        printf("Essential file(s) missing. Aborting...\n");
        return NULL;
    }

    EMBEDDING* model = createModel();
    int m, n;
    if (checkFileExists(embedding_filename))
    {
        getFileDimensions(embedding_filename, &m, &n);
        model->vocab_size = m;
        model->dimension = n-1;
        getEmbeddingParametersFromFile(model, embedding_filename);
    }

    getFileDimensions(X_filename, &m, &n);
    model->batch_size = m;

    model->X = getMatrixFromFile(model, X_filename);
    model->Y = getMatrixFromFile(model, Y_filename);
    model->W1 = getMatrixFromFile(model, W1_filename);
    model->W2 = getMatrixFromFile(model, W2_filename);
    model->b1 = getMatrixFromFile(model, b1_filename);
    model->b2 = getMatrixFromFile(model, b2_filename);
    return model;
}


void getEmbeddingParametersFromFile(EMBEDDING* model, char* filename)
{
    initialiseModelHashtable(model);
    FILE* fp = fopen(filename, "r");
    char* temp = (char*)malloc(sizeof(char)*INT_MAX);
    char *line, *token, *save, *ptr;
    char word[50];
    int pos = 0, dim = 0;
    while(fgets(temp, INT_MAX, fp))
    {
        line = trim(temp);
        token = strtok_r(line, ",", &save);
        strcpy(word, token);

        NODE* node = (NODE*)malloc(sizeof(NODE));
        node->word = (char*)malloc(sizeof(char)*strlen(word));
        strcpy(node->word, word);

        node->onehotvector = createOneHot(node, model);
        insert(node, model);

        node->wordvector = createZerosArray(1, model->dimension);
        token = strtok_r(NULL, ",", &save);
        dim = 0;
        while(token!=NULL)
        {
            double val = strtod(token, &ptr);
            node->wordvector[0][dim] = val;
            dim++;
            token = strtok_r(NULL, ",", &save);
        }
        free(line);
    }
    free(temp);
}

EMBEDDING* loadModelEmbeddings(char* embedding_filename)
{
    if(!checkFileExists(embedding_filename))
    {
        printf("File missing. Aborting...\n");
        return NULL;
    }
    EMBEDDING* model = createModel();
    int m, n;
    getFileDimensions(embedding_filename, &m, &n);
    model->vocab_size = m;
    model->dimension = n-1;
    getEmbeddingParametersFromFile(model, embedding_filename);
    return model;
}