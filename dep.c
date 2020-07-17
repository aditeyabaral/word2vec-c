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
    while(word[i])
        val+= pow(word[i], i+1);
    val%= vocab_size;
    return val;
}

void insert(NODE* node, EMBEDDING* model)
{
    int index = getHashvalue(node->word, model->vocab_size);
    while(model->hashtable[index]->mark == true)
    {
        if(!strcmp(model->hashtable[index]->word, node->word))
            return;
        index = (index+1)%model->vocab_size;
    }
    model->hashtable[index] = node;
    model->hashtable[index]->mark = true;
}

void init(EMBEDDING *model)
{
    model = (EMBEDDING*)malloc(sizeof(EMBEDDING));
    model->hashtable = NULL;
    model->context = 2;
    model->alpha = 0.01;
    model->context = 2;
    model->dimension = 100;
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

void display(float **a, int m, int n)
{
    for(int i = 0; i<m ;i++)
    {
        for(int j = 0; j<n; j++)
            printf("%f ", a[i][j]);
        printf("\n");
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

int getVocabularySize(char* corpus)
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
    //printf("Vocab: %s\n", temp2);
    return V;
}

void OneHotEncoding(EMBEDDING* model, char* corpus)
{
    char* cleaned_corpus = remove_punctuations(corpus);
    char* token=strtok(cleaned_corpus, " ");
	char word[30];
    model->vocab_size = getVocabularySize(cleaned_corpus);
    while (token != NULL)
    {
        strcpy(word, token);
		strcpy(word, trim(word));
        NODE* node = (NODE*)malloc(sizeof(NODE));
        strcpy(node->word, word);
        node->mark = true;
        insert(node, model);
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
    /*
    Get one hot vectors from OHV(), along with vocab_size.
    */
}