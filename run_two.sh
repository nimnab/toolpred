#!/bin/bash


source ~/.bashrc 
 
filename=$1
# shift
# tags=$*
#seeds=(15 896783 9 12 45234)
#orders=(62 3 2 1)

#seeds=(0.01 0.001 1e-5 1e-10 1e-20)
#orders=(100 0.1 1 10)

seeds=(0 1 2)
orders=(0 1 2)



for c in "${orders[@]}"
do
	for j in "${seeds[@]}"
	do
	    python3 -m sspace.hierachy.$filename ${j} ${c} &
	done
done