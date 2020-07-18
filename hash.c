#include "header.h"

int getHashvalue(char *word, int vocab_size)
{
    int val = 0;
    int i = 0;
    while(word[i] != '\0')
    {
        val+= word[i]*(i+1);;
        i++;
    }
    val%= vocab_size;
    return val;
}

void insert(NODE* node, EMBEDDING* model)
{
    int index = getHashvalue(node->word, model->vocab_size);
    while(model->hashtable[index] != NULL)
    {
        if(!strcmp(model->hashtable[index]->word, node->word))
            return;
        index = (index+1)%model->vocab_size;
        
    }
    model->hashtable[index] = node;
}

void createHashtable(EMBEDDING* model, char* corpus)
{
    model->vocab_size = getVocabularySize(model);
    initialiseModelHashtable(model);
    char word[30];
    char *save, *token;
    char *temp = (char*)malloc(sizeof(char)*strlen(model->vocab));
    strcpy(temp, model->vocab);
    token = strtok_r(temp, " ", &save);
    while (token != NULL)
    {
        strcpy(word, token);
        NODE* node = (NODE*)malloc(sizeof(NODE));
        node->word = (char*)malloc(sizeof(char)*strlen(word));
        strcpy(node->word, word);
        node->onehotvector = createOneHot(node, model);
        insert(node, model);
        token = strtok_r(NULL, " ", &save);
    }
}