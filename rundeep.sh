#!/bin/bash


source ~/.bashrc 
 
filename=$1
# shift
# tags=$*

 
seeds=(0 1 2 3 4 5)
 

for j in "${seeds[@]}"
do
    python3 -m sspace.$filename ${j}
    wait
done