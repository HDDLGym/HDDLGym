#!/bin/bash

set -e

seed=1
if ! [ -z $1 ]; then seed=$1 ; fi

cd $(dirname "$0")
mkdir -p hard

for nslots in $(seq 2 1 10) $(seq 12 2 20) $(seq 24 4 40) $(seq 45 5 80) ; do

    outfile="$(pwd)/hard/pb-${nslots}slots-seed${seed}.snake"
    echo $outfile
    python3 generate_combinatorial_problem.py $nslots $seed > $outfile

done
