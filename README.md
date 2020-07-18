# Word2Vec
Finding Distributed Representations of Words<br>
 
## Compilation

Convert into executable shell scripts with<br>

```
$chmod +x compile.sh run.sh execute.sh
```

To compile, use 
```
$ ./compile.sh
```
or 
```
$ gcc w2v.c dep.c preprocess.c hash.c disp.c mat.c -lm
```

Similarly to run, use

```
$ ./run.sh input_text_filename output_text_filename
```

```
$ ./a.out < sample2.txt > modelDetails.txt
```

To perform both compilation and execution, use

```
$ ./execution.sh input_text_filename output_text_filename
```
 
