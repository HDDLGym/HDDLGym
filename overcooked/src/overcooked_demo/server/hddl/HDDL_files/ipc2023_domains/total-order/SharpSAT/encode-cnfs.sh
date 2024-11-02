#!/bin/bash

for f in cnfs/*.cnf ; do 
    python3 cnf2hddl.py $f > problems/p-$(basename $f|sed 's/\.cnf$/.hddl/g')
done
