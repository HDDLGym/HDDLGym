#!/bin/bash

g++ -O2 generator.cpp

mkdir problems

i=3
./a.out 2 $((420 + $i)) $i 2 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 2 $((420 + $i)) $i 10 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))


./a.out 3 $((420 + $i)) $i 1 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))

./a.out 3 $((420 + $i)) $i 10 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 10 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 10 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 7 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 7 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 7 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 5 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 5 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 5 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 5 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 4 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 3 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 3 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 2 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 3 $((420 + $i)) $i 2 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))


./a.out 4 $((420 + $i)) $i 50 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 50 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 10 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 10 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 5 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 5 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 3 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 3 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 2 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))
./a.out 4 $((420 + $i)) $i 2 > problems/pfile$(printf "%02d" $i).hddl ; i=$((i+1))

