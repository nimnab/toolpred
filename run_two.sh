#!/bin/bash


source ~/.bashrc 
 
filename=$1
# shift
# tags=$*

 
seeds=(15 896783 9 12 45234)
orders=(62 3 2 1)



for c in "${orders[@]}"
do
	for j in "${seeds[@]}"
	do
	    python3 -m sspace.$filename ${j} ${c} &
	done
done