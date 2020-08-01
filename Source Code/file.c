#include "header.h"

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


EMBEDDING* loadModelForTraining(char* embedding_filename, char* X_filename, char* Y_filename, char* W1_filename, char* W2_filename)
{
    EMBEDDING* model = createModel();
    //todo
    return model;
}

void getEmbeddingParametersFromFile(EMBEDDING* model, char* filename)
{
    FILE* fp = fopen(filename, "r");
    char* tmp = (char*)malloc(sizeof(char)*INT_MAX);
    char *line, *token, *save;
    int vocab_size = 0, dimension = 0, len;
    fscanf(fp, "%[^EOF]s", tmp);
    char* cleaned = trim(tmp);
    //printf("%s\n", cleaned);
    token = strtok_r(cleaned, "\n", &save);
    while(token!=NULL)
    {
        vocab_size++;
        token = strtok_r(NULL, "\n", &save);
    }   
    printf("%d\n", vocab_size);
}

EMBEDDING* loadModelEmbeddings(char* embedding_filename)
{
    FILE* fp = fopen(embedding_filename, "r");
    if (fp == NULL)
    {
        printf("No file found.\n");
        return NULL;
    }
    fclose(fp);

    EMBEDDING* model = createModel();
    getEmbeddingParametersFromFile(model, embedding_filename);

    
    #if 0
    rewind(fp);
    int vocab_size = 0;
    char* token, *save;
    char word[50];    

    while(fgets(tmp, INT_MAX, fp))
    {
        line = trim(tmp);
        vocab_size++;
        token = strtok_r(line, ",", &save);
        strcpy(word, token);

        NODE* node = (NODE*)malloc(sizeof(NODE));
        node->word = (char*)malloc(sizeof(char)*strlen(word));
        strcpy(node->word, word);
        node->wordvector = NULL;
        //node->onehotvector = createOneHot(node, model);
        //insert(node, model);

    }
    free(line);
    #endif
    return model;
}