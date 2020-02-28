#!/bin/bash


source ~/.bashrc 
 
filename=$1
# shift
# tags=$*

 
seeds=(15 896783 9 12 45234)
classes=(1 3 5 7 9 11 13 15)



for c in "${classes[@]}"
do
	for j in "${seeds[@]}"
	do
	    python3 -m sspace.$filename ${j} ${c} &
	done
done