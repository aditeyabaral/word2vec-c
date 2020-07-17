#include "header.h"

float relu(float x)
{
    if (x>=0.0)
        return x;
    return 0.0;
}

float** softmax(float** M, int m, int n, int axis)
{
    /* Allocating space for output 2D matrix*/
    float** softmax_out = createZerosArray(m, n);

    /* Setting all values in output to be exp(corresponding value in input M) */
    for(int i=0; i<m; ++i)
    {
        for(int j=0; j<n; ++j)
        {
            softmax_out[i][j] = exp(M[i][j]);   
        }
    }

    /* Calculating column and row sums of output matrix */
    double* colsums = (double*)malloc(n*sizeof(double));
    double* rowsums = (double*)malloc(m*sizeof(double));
    memset(rowsums, 0, m*sizeof(double));
    memset(colsums, 0, n*sizeof(double));
    double sum = 0.0;
    for(int i=0; i<m; ++i)
    {
        for(int j=0; j<n; ++j)
        {
            rowsums[i] += softmax_out[i][j];
            colsums[j] += softmax_out[i][j];
            sum += softmax_out[i][j];
        }
    }
    if(axis==1)  /*Divide each element by its column sum*/
    {
        for(int i=0; i<m; ++i)
        {
            for(int j=0; j<n; ++j)
            {
                softmax_out[i][j] /= colsums[j];
            }
        }
    }
    else if(axis==0) /*Divide each element by its row sum*/
    {
        for(int i=0; i<m; ++i)
        {
            for(int j=0; j<n; ++j)
            {
                softmax_out[i][j] /= rowsums[i];
            }
        }
    }
    else /* Divide each element by matrix sum */
    {
        for(int i=0; i<m; ++i)
        {
            for(int j=0; j<n; ++j)
            {
                softmax_out[i][j] /= sum;
            }
        }
    }
    free(colsums);
    free(rowsums);
    return softmax_out;
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
    //displayHashtable(model);
    //model->hashtable[index]->mark = true;
}

EMBEDDING* initialiseModelParameters(char* corpus, int C, int N, float alpha)
{
    EMBEDDING* model = (EMBEDDING*)malloc(sizeof(EMBEDDING));
    int len = strlen(corpus);
    model->corpus_length = len;
    model->corpus = (char*)malloc(sizeof(char)*len);
    model->clean_corpus = (char*)malloc(sizeof(char)*len);
    char* cleaned_corpus = remove_punctuations(corpus);
    strcpy(model->clean_corpus, cleaned_corpus);
    strcpy(model->corpus, corpus);
    model->hashtable = NULL;

    if (C > 0)
        model->context = C;
    else
        model->context = 2;

    if (N > 0)
        model->dimension = N;
    else
        model->dimension = 100;

    if (alpha > 0)
        model->alpha = alpha;
    else
        model->alpha = 0.01;
    return model;
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

float** createZerosArray(int m, int n)
{
    float** array = (float**)malloc(sizeof(float*)*m);
    for(int i = 0; i<m ;i++)
    {
        array[i] = (float*)malloc(sizeof(float)*n);
        for(int j = 0; j<n; j++)
            array[i][j] = 0;
    }
    return array;
}

float** createOnesArray(int m, int n)
{
    float** array = (float**)malloc(sizeof(float*)*m);
    for(int i = 0; i<m ;i++)
    {
        array[i] = (float*)malloc(sizeof(float)*n);
        for(int j = 0; j<n; j++)
            array[i][j] = 1;
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

float** multiply(float **M1, float **M2, int m1, int n1, int m2, int n2)
{
    float **result = createZerosArray(m1, n2);
    displayArray(result, m1, n2);
    for(int i = 0; i<m1 ;i++)
    {
        for(int j=0; j<n2 ;j++)
        {
            for(int k=0; k<m2; k++)
                result[i][j]+= M1[i][k] + M2[k][j];
        }
    }
    return result;
}

void displayHashtable(EMBEDDING* model)
{
    for(int i=0;i<model->vocab_size;i++)
    {
        if(model->hashtable[i] != NULL)
        {
            printf("%s\n", model->hashtable[i]->word);
            for(int j=0; j<model->vocab_size; ++j)
                printf("%.1f ", model->hashtable[i]->onehotvector[0][j]);
            printf("\n\n");
        }
    }
}

void displayModel(EMBEDDING* model)
{
    printf("Input: %s\n\n", model->corpus);
    printf("Cleaned Text : %s\n\n", model->clean_corpus);
    printf("Vocabulary: %s\n\n", model->vocab);
    printf("C: %d\nN: %d\nalpha: %f\nVocab Size: %d\n\n", model->context, model->dimension, model->alpha, model->vocab_size);
    printf("Hashtable: \n");
    displayHashtable(model);
    printf("\nW1: \n");
    displayArray(model->W1, model->dimension, model->vocab_size);
    printf("\nW2: \n");
    displayArray(model->W2, model->vocab_size, model->dimension);
    printf("\nb1: \n");
    displayArray(model->b1, model->dimension, 1);
    printf("\nb2: \n");
    displayArray(model->b2, model->vocab_size, 1);

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

int getVocabularySize(EMBEDDING* model)
{
    char* corpus = model->clean_corpus;
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
    free(temp1);
    free(temp3);
    model->vocab = temp2;
    return V;
}

float** createOneHot(NODE* node, EMBEDDING* model)
{
    int index = getHashvalue(node->word, model->vocab_size);
    while(model->hashtable[index] != NULL)
    {
        if(!strcmp(model->hashtable[index]->word, node->word))
            break;
        index = (index+1)%model->vocab_size;
    }
    float** oneHotVector = createZerosArray(1, model->vocab_size);
    if(oneHotVector == NULL)
    {
        printf("Failed to allocate memory for one hot vector!\n");
        return NULL;
    }
    oneHotVector[0][index] = 1;
    return oneHotVector;
}

void createHashtable(EMBEDDING* model, char* corpus)
{
    model->vocab_size = getVocabularySize(model);
    initialiseModelHashtable(model);
    //printf("Vocabulary: %s\nSize: %d\n", model->vocab, model->vocab_size);
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

void createXandY(EMBEDDING* model)
{
    int num_words = 0;
    char* token1 , *save1;
    char* temp1 = (char*)malloc(sizeof(char)*model->corpus_length);
    strcpy(temp1, model->clean_corpus);
    token1 = strtok_r(temp1, " ", &save1);
    while(token1 != NULL)
    {
        num_words++;
        token1 = strtok_r(NULL, " ", &save1);
    }
    if ((2*model->context+1) >= num_words)
    {
        printf("Not enough words available for context. Window >= number of words.\n");
        return;
    }
    int ctr1 = 1, ctr2 = 1;
    char* temp2 = (char*)malloc(sizeof(char)*model->corpus_length);
    strcpy(temp1, model->clean_corpus);
    token1 = strtok_r(temp1, " ", &save1);
    char* token2, *save2;

    //int m = 1+(num_words%(2*model->context+1)); //Number of possible examples = X and y have 'm' columns
    char* X_words = (char*)malloc(sizeof(char)*INT_MAX);
    char* y_words = (char*)malloc(sizeof(char)*INT_MAX);

    while(token1 != NULL && ctr1 <= (num_words-2*model->context))
    {
        ctr2 = 1;
        strcpy(temp2, model->clean_corpus);
        token2 = strtok_r(temp2, " ", &save2);
        while(token2 != NULL && ctr2!=ctr1)
        {
            token2 = strtok_r(NULL, " ", &save2);
            ctr2++;
        }
        for(int i=1;i<=model->context;i++)
        {
            strcat(X_words, token2);
            strcat(X_words, " ");
            token2 = strtok_r(NULL, " ", &save2);
        }
        strcat(y_words, token2);
        strcat(y_words, "\n");
        token2 = strtok_r(NULL, " ", &save2);
        for(int i=1;i<=model->context;i++)
        {
            strcat(X_words, token2);
            strcat(X_words, " ");
            token2 = strtok_r(NULL, " ", &save2);
        }
        strcat(X_words, "\n");
        ctr1++;
        token1 = strtok_r(NULL, " ", &save1);
    }

    int m = ctr1-1;
    printf("Input: \n");
    printf("%s\n\n", model->clean_corpus);
    printf("m: %d\n\n", m);
    printf("X: \n");
    printf("%s\n\n", X_words);
    printf("y: \n");
    printf("%s\n\n", y_words);

}

void train(char* corpus, int C, int N, float alpha, int random_state)
{
    EMBEDDING* model = initialiseModelParameters(corpus, C, N, alpha);
    createHashtable(model, corpus);
    model->W1 = createArray(model->dimension, model->vocab_size, random_state);
    model->W2 = createArray(model->vocab_size, model->dimension, random_state);
    model->b1 = createArray(model->dimension, 1, random_state);
    model->b2 = createArray(model->vocab_size, 1, random_state);
    //displayModel(model);
    createXandY(model);
}
