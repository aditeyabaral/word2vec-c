# Word2Vec
Finding Distributed Representations of Words<br>
Note - The project is not complete and is a work in progress. Once finished all changes will be moved to [NLPC](https://github.com/aditeyabaral/NLPC).

Stuff that works: <br>
* input corpus text preprocessing
* creation of context vectors (CBOW model) and prediction word vectors (X and y respectively)
* forward propagation in neural network

Stuff we aren't sure that might work: <br>
* back propagation
* cost function, softmax implementation

Remember to redirect output to a text file after you execute. The application quits after training.
 
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
$ gcc w2v.c dep.c preprocess.c hash.c disp.c mat.c -lm
```

Followed by

```
$ ./a.out < textfilename 
```


