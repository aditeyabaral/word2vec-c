#include "header.h"

double** getVector(EMBEDDING* model, char* word)
{
    int index = getHashvalue(word, model->vocab_size);
    int ctr = 0;
    while(model->hashtable[index] != NULL && ctr != model->vocab_size)
    {
        if(!strcmp(model->hashtable[index]->word, word))
            return model->hashtable[index]->wordvector;
        index = (index+1)%model->vocab_size;
        ctr+= 1;
    }
    return NULL;
}

char* getWord(EMBEDDING* model, double** vector)
{
    char* word = (char*)malloc(sizeof(char)*50);
    double max_sim = -1.0, sim;
    for(int i = 0; i < model->vocab_size; i++)
    {
        sim = cosine_similarity(vector, model->hashtable[i]->wordvector, model->dimension);
        if (sim > max_sim)
        {
            max_sim = sim;
            memset(word, '\0', 50);
            strcpy(word, model->hashtable[i]->word);
        }
    }
    return word;
}