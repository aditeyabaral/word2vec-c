#include "header.h"

double similarity(EMBEDDING* model, char* word1, char* word2)
{
    double** v1 = getWordVector(model, word1);
    double** v2 = getWordVector(model, word2);
    return dot(v1, v2, model->dimension)/(norm(v1, 1, model->dimension)*norm(v2, 1, model->dimension));
}

double** getWordVector(EMBEDDING* model, char* word)
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

