# Word2Vec-C
Implementation of Finding Distributed Representations of Words and Phrases and their Compositionality as in the original [Word2Vec Research Paper](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality) by Tomas Mikolov.<br>

This implementation has been built using the C programming language and uses the Continuous-Bag-Of-Words Model (CBOW) over the Skip-Gram model as put forward in the paper. It includes all the basic functionalities supported by the [gensim](https://radimrehurek.com/gensim/models/word2vec.html) python library and also allows users to implement other higher level functions using these building blocks.<br>

The implementation was built from scratch, from the text pre-processing to the neural network. Although the speed of execution made up for a non-implementable vectorized approach, the model requires a lot of hyperparameter tuning to obtain best results. It is advised to attempt execution on a smaller corpus, before trying a larger one.<br><br>

Note - All changes will also be pushed to [NLPC](https://github.com/aditeyabaral/NLPC).

## How does it work?

## Compilation

To make compilation easy, a simple shell script has been included. Run the following commands:<br>
```
$ chmod +x compile.sh
$ ./compile.sh
```

Alternatively, compile all the source code files using <br>
```
$ gcc w2v.c dep.c preprocess.c hash.c disp.c mat.c file.c neuralnetwork.c func.c mem.c -lm
```

## Execution

Execute with input redirection, with the one command line argument being the path to the corpus.<br>
```
$ ./a.out < path_to_corpus 
```

## Supported Functionalities

### Cosine Similarity 
To find the cosine similarity between two words, use
```
double sim = similarity(model, word1, word2);
```
### Cosine Distance
To find the cosine distance between two words, use 
```
double dist = distance(model, word1, word2);
```
### Extract Embeddings 
To find a word's embedding, use
```
double** vector = getVector(model, word);
```
### Most Similar Word by Vector 
To obtain the word most similar to a vector, use
```
char* word = getWord(model, vector);
```
### K Most Similar Words by Word 
To obtain a set of K words most similar to a given word (in decreasing order of similarity), use
```
char* similar_words = mostSimilarByWord(model, word, k);
```
### K Most Similar Words by Word
To obtain a set of K words most similar to a given vector (in decreasing order of similarity), use
```
char* similar_words = mostSimilarByVector(model, word, k);
```

To support other functionalities like vector operations between embeddings, miscellaneous matrix operations have been added as well. More information about them can be found under Matrix Utilities in the header file.

## Coming Soon

* loading model from file
    * support load and use
    * support load and train
