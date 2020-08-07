# Word2Vec-C
Implementation of Finding Distributed Representations of Words and Phrases and their Compositionality as in the original [Word2Vec Research Paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality) by Tomas Mikolov.<br>

This implementation has been built using the C programming language and uses the Continuous-Bag-Of-Words Model (CBOW) over the Skip-Gram model as put forward in the paper. It includes all the basic functionalities supported by the [gensim](https://radimrehurek.com/gensim/models/word2vec.html) python library and also allows users to implement other higher level functions using these building blocks.<br>

The implementation was built from scratch, from the text pre-processing to the neural network. Although the speed of execution made up for a non-implementable vectorized approach, the model requires a lot of hyperparameter tuning to obtain best results. It is advised to attempt execution on a smaller corpus, before trying a larger one.<br><br>

Note - All changes will also be pushed to [NLPC](https://github.com/aditeyabaral/NLPC).

# How does it work?

## Using the Library

To use Word2Vec-C, the simplest way is to include the ```word2vec.h``` header file in your code. 

```sh
#include <stdio.h>
#include "word2vec.h"
int main(){
    ...
    ...
}
```

Compile your code with ```-lm``` to link with the required ```math.h``` header file. <br>

```sh
$ gcc file.c -lm
```

Alternatively, you can also compile all the source code files included.

## Compiling from Source Code

To compile Word2Vec-C with your source code, compile your files with all the dependency files. Replace ```w2v.c``` with your file(s). For easy compilation, a simple shell script has been included. Run the following commands:<br>

```sh
$ chmod +x compile.sh
$ ./compile.sh
```

Alternatively, compile all the source code files using <br>
```sh
$ gcc w2v.c dep.c preprocess.c hash.c disp.c mat.c file.c neuralnetwork.c func.c mem.c -lm
```

# Using Word2Vec-C

To start using Word2Vec-C in your code, first create or load the model with one of the following instructions

## Initialise Model
Use this to create an empty model to train from scratch.

```sh
EMBEDDING* model = createModel();
```

## Load Model Embeddings
Use this to load only the model's embeddings

```sh
EMBEDDING* model = loadModelEmbeddings("model-embeddings.csv");
```

## Load Model
Use this to load the entire model - X, y, weights and bias

```sh
EMBEDDING* model = loadModelForTraining("model-embeddings.csv", "model-X.csv", "model-y.csv", 
                "model-weights-w1.csv", "model-weights-w2.csv", "model-bias-b1.csv", "model-bias-b2.csv");
```

Now you can either train the model (if model has only been initialised) or used as needed. Remember to call ```destroyModel(model)``` to free the model after use.

More information about loading as well as saving models can be found at the end of this README.

# Supported Functionalities

### Train Model

To train the model, use
```sh
train(model, corpus, context_window, embedding_dimension, alpha, epochs, random_state, save_model_corpus);
```

### Cosine Similarity 
To find the cosine similarity between two words, use
```sh
double sim = similarity(model, word1, word2);
```
### Cosine Distance
To find the cosine distance between two words, use 
```sh
double dist = distance(model, word1, word2);
```
### Extract Embeddings 
To find a word's embedding, use
```sh
double** vector = getVector(model, word);
```
### Most Similar Word by Vector 
To obtain the word most similar to a vector, use
```sh
char* word = getWord(model, vector);
```
### K Most Similar Words by Word 
To obtain a set of K words most similar to a given word (in decreasing order of similarity), use
```sh
char* similar_words = mostSimilarByWord(model, word, k);
```
### K Most Similar Words by Vector
To obtain a set of K words most similar to a given vector (in decreasing order of similarity), use
```sh
char* similar_words = mostSimilarByVector(model, vector, k);
```

# Saving and Loading Models

A model can be saved and loaded for further training as well as to extract embeddings for use. 

## Saving a Model
To save the model and its embeddings, use

```sh
saveModel(model, save_corpus);
```
The ```save_corpus``` argument is used to save the corpus used to train and takes in boolean values.

## Loading a Model

### Load Model from Embeddings
To load a model's embeddings, ensure that the embedding CSV file contains data in the following format

| word1 | embedding1 | embedding2 | ... | embeddingN |
|-------|------------|------------|-----|------------|
| word2 | embedding1 | embedding2 | ... | embeddingN |
| ...   | ...        | ...        | ... | ...        |
| wordV | embedding1 | embedding2 | ... | embeddingN |

Load the model using
```sh
EMBEDDING* model = loadModelEmbeddings("model-embeddings.csv");
```

Note that this function will allow you to only use the embeddings and their associated functions. It does not support further training of the model. 

### Load Model from Weights, Biases, X and y
To load a model for further training and usage, use 

```sh
EMBEDDING* model = loadModelForTraining("model-embeddings.csv", "model-X.csv", "model-y.csv", 
                    "model-W1.csv", "model-W2.csv", "model-b1.csv", "model-b2.csv");
```
The first argument - the embeddings file can be left ```NULL```. 


To support other functionalities like vector operations between embeddings, miscellaneous matrix operations have been added as well. More information about them can be found under ```MATRIX UTILITIES``` in the header file.
