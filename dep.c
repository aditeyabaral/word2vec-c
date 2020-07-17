#include "header.h"

float relu(float x)
{
    if (x>=0.0)
        return x;
    return 0.0;
}

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
    //model->hashtable[index]->mark = true;
}

void initialiseModelParameters(EMBEDDING *model)
{
    model = (EMBEDDING*)malloc(sizeof(EMBEDDING));
    model->hashtable = NULL;
    model->context = 2;
    model->alpha = 0.01;
    model->context = 2;
    model->dimension = 100;
}

void initialiseModelHashtable(EMBEDDING* model)
{
    model->hashtable = (NODE**)malloc(sizeof(NODE*)*model->vocab_size);
    for(int i = 0; i<model->vocab_size; i++)
        model->hashtable[i] = NULL;
}

float** createArray(int m, int n, int random_state)
{
    srand(random_state);
    float** array = (float**)malloc(sizeof(float*)*m);
    for(int i = 0; i<m ;i++)
    {
        array[i] = (float*)malloc(sizeof(float)*n);
        for(int j = 0; j<n; j++)
            array[i][j] = (float)rand()/(float)(RAND_MAX);
    }
    return array;
}

void displayArray(float **a, int m, int n)
{
    for(int i = 0; i<m ;i++)
    {
        for(int j = 0; j<n; j++)
            printf("%f ", a[i][j]);
        printf("\n");
    }
}

void displayHashtable(EMBEDDING* model)
{
    for(int i=0;i<model->vocab_size;i++)
    {
        if(model->hashtable[i] != NULL)
            printf("%s\n", model->hashtable[i]->word);
    }
}

char* remove_punctuations(char *sent)
{
	char punc[] = {'!', ']', '{', '#', '.', '<', '/', '(', '~', ',', '%', ';', '`', ':', '?', '+', '$', '^', '\\', '@', '*', '}', '=', '_', '\"', ')', '-', '|', '[', '&', '>'};
	int check, ctr = 0;
	char *s = (char*)malloc(sizeof(char)*strlen(sent));
	for (int i = 0; i< strlen(sent); i++)
	{
		check = 0;
		for (int j = 0; j< 30; j++)
		{
			if ((char)sent[i] == (char)punc[j])
			{
				check = 1;
				break;
			}
		}
		if (check==0)
			s[ctr] = sent[i];
        else
            s[ctr] = ' ';	
        ctr++;
	}
	s[ctr]='\0';	
	return s;
}

char* trim(char* word)
{
    char* s = (char*)malloc(sizeof(char)*50);
    int pos1 = 0, pos2 = 0;
    while(word[pos1]==' ')
        pos1++;
    while(word[pos1]!=' ' && word[pos1]!='\0')
    {
        s[pos2] = word[pos1];
        pos1++;
        pos2++;
    }
    s[pos2] = '\0';
    return s;
}

int getVocabularySize(EMBEDDING* model, char* corpus)
{
    int len = strlen(corpus);
    char* temp1 = (char*)malloc(sizeof(char)*len);
    char* temp2 = (char*)malloc(sizeof(char)*len);
    char* temp3 = (char*)malloc(sizeof(char)*len);
    strcpy(temp1, corpus);
    strcpy(temp2, "");
    strcpy(temp3, "");
    char* token1;
    char* token2;
    char* save1;
    char* save2;
    int V = 0;
    
    token1 = strtok_r(temp1, " ", &save1);
    while(token1 != NULL)
    {
        strcpy(temp3, "");
        strcpy(temp3, temp2);        
        token2 = strtok_r(temp3, " ", &save2);
        bool flag = false;
        while(token2!=NULL)
        {
            if (!strcmp(token1, token2))
            flag = true;
            token2 = strtok_r(NULL, " ", &save2);
        }
        if (!flag)
        {
            strcat(temp2, token1);
            strcat(temp2, " ");
            V++;
        }
        token1 = strtok_r(NULL, " ", &save1);
    }
    model->vocab = temp2;
    return V;
}

void createHashtable(EMBEDDING* model, char* corpus)
{
    char* cleaned_corpus = remove_punctuations(corpus);
    model->vocab_size = getVocabularySize(model, cleaned_corpus);
    initialiseModelHashtable(model);
    printf("Vocabulary: %s\nSize: %d\n", model->vocab, model->vocab_size);
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
        insert(node, model);
        token = strtok_r(NULL, " ", &save);
    }
}

void train(EMBEDDING* model, int C, int N, float alpha, char* corpus)
{
    if (C > 0)
        model->context = C;
    if (N > 0)
        model->dimension = N;
    if (alpha > 0)
        model->alpha = alpha;
    createHashtable(model, corpus);
    displayHashtable(model);
    /*
    Get one hot vectors from OHV(), along with vocab_size.
    */
}