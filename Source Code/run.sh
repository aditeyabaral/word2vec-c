#!/bin/sh
if [ $1 -eq 0 ]
then
    ./a.out $1 < $2

else
    ./a.out $1
fi