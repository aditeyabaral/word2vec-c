# Word2Vec
Finding Distributed Representations of Words<br>
Note - The project is not complete and is a work in progress. Once finished all changes will be moved to [NLPC](https://github.com/aditeyabaral/NLPC).


Model gets saved automatically, along with parameters.
Documentation for functions will be added soon :)<br>

## Validation
Stuff that works: <br>
* input corpus text preprocessing
* creation of context vectors for CBOW model (X) and prediction word vectors (y)
* matrix operations for neural network architecture
* forward propagation
* back propagation
* gradient descent
* saving of model

Stuff we need to verify but *most likely* working<br>
* cost function
* Extraction of embeddings

## Compilation

Convert into executable shell scripts with<br>

```
$chmod +x compile.sh
```

To compile and run, use 
```
$ ./compile.sh
```
or 
```
$ gcc w2v.c dep.c preprocess.c hash.c disp.c mat.c file.c neuralnetwork.c -lm
```

## Execution

Execute with

```
$ ./a.out < corpus_textfilename 
```

## Supported Functionalities (Work in Progress)

To find the cosine similarity between two words, use
```
double sim = similarity(model, word1, word2);
```

To find the cosine distance, use 
```
double dist = distance(model, word1, word2);
```

A word's embedding can be extracted with
```
double** vector = getVector(model, word);
```

Similary, to obtain the word most similar to a vector, use
```
char* word = getWord(model, vetor);
```

More support coming soon!


## To-Do

* loading model from file
    * support load and use
    * support load and train
* w2v functions like similarity, most_similar etc
