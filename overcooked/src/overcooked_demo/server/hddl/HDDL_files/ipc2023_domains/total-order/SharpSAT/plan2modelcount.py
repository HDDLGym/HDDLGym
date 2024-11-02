
import sys
import re
import math

cnffile = sys.argv[1]
if len(sys.argv) > 2:
    planfile = open(sys.argv[2], 'r')
else:
    planfile = sys.stdin

nvars = 0
for line in open(cnffile, 'r').readlines():
    if line.startswith('p'):
        nvars = int(line.rstrip().split(" ")[2])
        break
print(f"{nvars} vars in CNF")

planbegun = False
nmodels = 0
for line in planfile.readlines():
    if not planbegun:
        if "==>" in line:
            planbegun = True
        continue
    if "<==" in line:
        break
    
    match = re.search(r'OUTPUT_EXPONENTIAL_COUNT n([0-9]+)', line)
    if match:
        depth = int(match.group(1))
        newmodels = int(math.pow(2, nvars-depth))
        print(f"SAT at depth {depth} => {newmodels} models")
        nmodels += newmodels

print(f"{nmodels} models found.")
    
