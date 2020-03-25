#!/bin/bash


source ~/.bashrc 
 
filename=$1
# shift
# tags=$*

 
#seeds=(15 896783 9 12 45234)
seeds=(0 1 2 3 4)


for j in "${seeds[@]}"
do
    python3 sspace/hierachy/$filename ${j} &
done