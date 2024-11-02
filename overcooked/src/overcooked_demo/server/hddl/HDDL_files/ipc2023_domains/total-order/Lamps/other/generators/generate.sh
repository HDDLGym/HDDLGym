#!/bin/bash

g++ -O2 generator.cpp

mkdir problems

i=1
./a.out 1 $((420 + $i)) $i 1 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 2 $((420 + $i)) $i 1 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 2 $((420 + $i)) $i 2 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 1 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 2 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 3 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 4 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 5 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 2 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 4 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))

##

./a.out 4 $((420 + $i)) $i 6 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 8 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 5 $((420 + $i)) $i 4 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 5 $((420 + $i)) $i 8 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 5 $((420 + $i)) $i 10 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 7 $((420 + $i)) $i 8 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 7 $((420 + $i)) $i 10 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 7 $((420 + $i)) $i 15 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 10 $((420 + $i)) $i 10 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 10 $((420 + $i)) $i 15 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))

##

./a.out 13 $((420 + $i)) $i 10 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 13 $((420 + $i)) $i 15 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 13 $((420 + $i)) $i 20 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 17 $((420 + $i)) $i 15 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 17 $((420 + $i)) $i 20 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 17 $((420 + $i)) $i 25 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 20 $((420 + $i)) $i 20 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 20 $((420 + $i)) $i 25 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 20 $((420 + $i)) $i 30 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
./a.out 25 $((420 + $i)) $i 30 > problems/pfile$(printf "%02d" $i).pddl ; i=$((i+1))
