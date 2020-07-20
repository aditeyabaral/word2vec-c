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

EMBEDDING* loadModel(char* embedding_filename)
{
    EMBEDDING* model = createModel();
    //todo
    return model;
}