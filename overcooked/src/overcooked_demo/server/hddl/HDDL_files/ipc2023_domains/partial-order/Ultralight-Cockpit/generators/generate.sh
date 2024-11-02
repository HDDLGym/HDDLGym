#!/bin/bash

LS="1 2 2 3 4 5 6 7 8 10 11 12 13 14 15 16 17 18 19 20"

mkdir problems

for p in $(seq 20) ; do
	PREF=$(printf "%02d" $p)
	L=$(echo $LS | cut -d' ' -f $p)
	echo $L
	python3 prob_instance_generator.py --num_landingSpots $L problems/pfile$PREF.hddl
done
