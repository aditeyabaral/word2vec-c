#include "header.h"

char* remove_punctuations(char *sent)
{
	char punc[] = {'\n', '!', ']', '{', '#', '.', '<', '/', '(', '~', ',', '%', ';', '`', ':', '?', '+', '$', '^', '\\', '@', '*', '}', '=', '_', '\"', ')', '-', '|', '[', '&', '>'};
	int check, ctr = 0;
	char *s = (char*)malloc(sizeof(char)*strlen(sent));
	for (int i = 0; i< strlen(sent); i++)
	{
		check = 0;
		for (int j = 0; j< 31; j++)
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
    int len = strlen(word);
    char* s = (char*)malloc(sizeof(char)*len);
    int pos1 = 0, pos2 = len-1, ctr = 0;
    while(word[pos1]==' ')
        pos1++;
    while(word[pos2]==' ')
        pos2--;
    while(pos1<=pos2)
    {
        s[ctr] = word[pos1];
        pos1++;
        ctr++;
    }
    s[pos1] = '\0';
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
    int len2 = strlen(temp2);
    temp2[len2-1] = '\0';
    model->vocab = (char*)malloc(sizeof(char)*len2);
    strcpy(model->vocab, temp2);
    free(temp1);
    free(temp2);
    free(temp3);
    return V;
}