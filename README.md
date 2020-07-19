# Word2Vec
Finding Distributed Representations of Words<br>
Note - The project is not complete and is a work in progress. Once finished all changes will be moved to [NLPC](https://github.com/aditeyabaral/NLPC).

Documentation for functions will be added soon :)<br>
Model gets saved automatically, along with parameters

## Validation
Stuff that works: <br>
* input corpus text preprocessing
* creation of context vectors for CBOW model (X) and prediction word vectors y)
* matrix operations for neural network architecture
* forward propagation
* back propagation
* gradient descent
* saving of model

Stuff we need to verify but *most likely* working<br>
* cost function
* Extraction of embeddings

## Compilation & Execution

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
$ gcc w2v.c dep.c preprocess.c hash.c disp.c mat.c -lm
```

Followed by

```
$ ./a.out < textfilename 
```

## To-Do

* reduce memory leaks
* loading model from file
    * support load and use
    * support load and train
* w2v functions like similarity, most_similar etc
