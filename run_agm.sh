#!/bin/bash


source ~/.bashrc 
 
filename=$1
# shift
# tags=$*

 
seeds=(15 896783 9 12 45234)
classes=(10 50 100 150 200 300)



for c in "${classes[@]}"
do
	for j in "${seeds[@]}"
	do
	    python3 -m sspace.$filename ${j} ${c} &
	done
done