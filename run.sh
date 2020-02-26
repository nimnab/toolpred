#!/bin/bash


source ~/.bashrc 
 
filename=$1
# shift
# tags=$*

 
seeds=(15 896783 9 322 45234)
 

for j in "${seeds[@]}"
do
    python3 -m sspace.$filename ${j} &
done