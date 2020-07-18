# Word2Vec
Finding Distributed Representations of Words<br>
 
## Compilation

Convert into executable shell scripts with<br>

```
$chmod +x compile.sh run.sh execute.sh
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
 
