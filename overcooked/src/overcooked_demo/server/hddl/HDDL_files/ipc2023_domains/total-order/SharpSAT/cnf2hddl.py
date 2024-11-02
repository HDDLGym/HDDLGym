
import sys

if len(sys.argv) < 2:
    print(f"Usage: {sys.argv[0]} <cnf-file>")
    exit(0)

cnffile = sys.argv[1]

clauses = []
nvars = 0
occlist = dict()
nunitcls = 0

for line in open(cnffile, 'r').readlines():
    words = line.rstrip().split(' ')
    if words[0] == 'p':
        # header
        continue
    if words[0].startswith('c'):
        # comment
        continue
    if len(words) <= 1:
        # empty line
        continue
    clause = [int(x) for x in words[0:-1]]
    for lit in clause:
        var = abs(lit)
        nvars = max(nvars, var)
        if lit not in occlist:
            occlist[lit] = []
        occlist[lit] += [len(clauses)]
    if len(clause) == 1:
        nunitcls += 1
    clauses += [clause]

maxnum = max(nvars, len(clauses)) + 2

objects = dict()
facts = []

facts += [f"CLAUSE_COUNT n{len(clauses)}"]
facts += ["satisfied_clause_count n0"]
facts += [f"unit_clause_count n{nunitcls}"]

objects["number"] = []
for i in range(3, maxnum+1):
    objects["number"] += [f"n{i}"]
for i in range(1, maxnum+1):
    facts += [f"SUCCESSOR n{i-1} n{i}"]

objects["literal"] = []
for i in range(1, nvars+1):
    objects["literal"] += [f"l{i}p", f"l{i}n"]
    facts += [f"OPPOSITE l{i}p l{i}n", f"OPPOSITE l{i}n l{i}p"]

objects["clause"] = []
for i in range(len(clauses)):
    c = clauses[i]
    objects["clause"] += [f"c{i}"]
    facts += [f"clause_length c{i} n{len(c)}"]
    for lit in c:
        var = abs(lit)
        litstr = f"l{var}p" if lit>0 else f"l{var}n"
        facts += [f"LIT_IN_CLAUSE {litstr} c{i}"]

for var in range(1, nvars+1):
    for lit in [var, -var]:
        litstr = f"l{var}p" if lit>0 else f"l{var}n"
        if lit not in occlist:
            occlist[lit] = []
        facts += [f"OCCURRENCE_COUNT {litstr} n{len(occlist[lit])}"]
        for oidx in range(len(occlist[lit])):
            cidx = occlist[lit][oidx]
            facts += [f"OCCURRENCE_INDEX {litstr} c{cidx} n{oidx+1}"]

print()
print("; #SAT (model counting) problem")
print("; Generated via cnf2hddl.py by D. Schreiber")
print()
print(f"; Encoded CNF: {cnffile}")
for line in open(cnffile, 'r').readlines():
    print(f"; {line.rstrip()}")
print()
friendly_cnf_name = cnffile.replace(".", "-").replace("/", "-")
print(f"(define (problem sharpsat-{friendly_cnf_name})")
print("(:domain sharpsat)")
#print("(:requirements :typing :hierachie)")
print("(:objects")
for sort in objects:
    if objects[sort]:
        print("   ", " ".join(objects[sort]), "-", sort)
print(")")
print("(:htn :parameters () :ordered-subtasks (and")
print("    (task1  (count-models n0))")
print("))")
print("(:init")
for fact in facts:
    print("   ", "(" + fact + ")")
print(")")
print("(:goal (and ))")
print(")")
