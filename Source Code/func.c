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

char* get_top_k_by_vector(EMBEDDING* model, double** vector, int k)
{
    SIM_INFO* sims = (SIM_INFO*)malloc(sizeof(SIM_INFO)*model->vocab_size);
    for(int i=0; i<model->vocab_size; ++i)
    {
        sims[i].word = (char*)malloc(sizeof(char)*50);
        strcpy(sims[i].word, model->hashtable[i]->word);
        sims[i].sim = cosine_similarity(model->hashtable[i]->wordvector, vector, model->dimension);
    }
    char* temp = (char*)malloc(sizeof(char)*50);
    double s;

    char* top_k_words = (char*)malloc(sizeof(char)*k*50);
    strcpy(top_k_words, "");
    for(int i=0; i<k; ++i)
    {
        for(int j=0; j<model->vocab_size; ++j)
        {
            if(sims[j].sim > sims[j+1].sim)
            {
                strcpy(temp, sims[j].word);
                strcpy(sims[j].word, sims[j+1].word);
                strcpy(sims[j+1].word, temp);

                s = sims[j].sim;
                sims[j].sim = sims[j+1].sim;
                sims[j+1].sim = s;
            }
        }
    }

    for(int t = model->vocab_size - k -1; t < model->vocab_size; ++t)
    {
        strcat(top_k_words, sims[t].word);
        strcat(top_k_words, "\n");
    }

    for(int i=0; i<model->vocab_size; ++i)
    {
        free(sims[i].word);
    }
    free(sims);
    free(temp);
    return top_k_words;
}

char* get_top_k_by_word(EMBEDDING* model, char* word, int k)
{
    double** word_vector = getVector(model, word);
    if(word_vector == NULL)
    {
        printf("%s not in vocabulary!\n", word);
        return NULL;
    }
    else
    {
        char* top_k_words = get_top_k_by_vector(model, word_vector, k);
        free2D(word_vector, 1, model->dimension);
        return top_k_words;
    }
}