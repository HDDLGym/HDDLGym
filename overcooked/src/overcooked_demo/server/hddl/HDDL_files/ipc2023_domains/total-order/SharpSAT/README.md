 
# SharpSAT HDDL domain

The SharpSAT domain models the problem #SAT. This problem is to compute the number of 
different variable assignments ("models") which satisfy a given propositional formula.
Another term for the problem is (exact) model counting. 

## Generating benchmarks

The encoder script `cnf2hddl.py` associated with `domain.hddl` can take SAT instances in
the DIMACS CNF format. Assumptions are: (a) There is no empty clause in the input. 
(b) There is no clause in the input which features a particular variable more than once
(rendering the clause tautologic or partially redundant). The script then creates a
HDDL problem file corresponding to the exact model counting problem for that formula.

```
python3 cnf2hddl.py <cnf-file> > <hddl-problem-file>
```

Arbitrarily difficult benchmarks can be generated, e.g., by using [benchmarks from the
International SAT Competitions](https://satcompetition.github.io/2022/downloads.html)
or generating random difficult 3-SAT instances with `generate_3sat.py <nvars> [<seed>]`.

## Interpreting plans

A found plan can be transformed in linear time with the associated decoder script. This
linear-time procedure just looks for specific actions `A_OUTPUT_EXPONENTIAL_COUNT d` and
adds $2^{|V| - d}$ to a model counter for each such action.

    python3 plan2modelcount.py <cnf-file> <plan-output>

Or, as a single call together with planning:

    ./my-htn-planner domain.hddl <hddl-problem-file> | python3 plan2modelcount.py <cnf-file>

## Background

The complexity class of #SAT, named #P, is (handwavingly) somewhere between NP and PSPACE 
and therefore not considered as hard as HTN planning. Nevertheless, a hierarchical 
planning model for #SAT is appealing due to the problem's natural hierarchical structure,
its comparably simple formulation, and a number of interesting search properties (see below).

We express instances of #SAT as TOHTN planning instances. The hierarchical model is 
based on the straight forward recursive CDP algorithm by Birnbaum and Lozinskii in "The 
Good Old Davis-Putnam Procedure Helps Counting Models" (JAIR '99), which is a modified 
DPLL search procedure. When finding a model at decision depth d, the procedure does not
terminate (as DPLL) but instead adds $2^{|V| - d}$ to its global model count and 
continues its search. We implemented propagation and backtracking of variable decisions
deterministically such that the only decisions a planner can make is choosing a literal
to branch on (or, if unit clauses are present, which one to satisfy first). The domain
does NOT have any dead-ends (DPLL backtracking is performed with explicit subtasks); 
however, the branching choices a planner makes can result in substantial differences 
with respect to the effective search space size. Therefore, informative heuristics and/
or restarts with different decision-making have the potential to make a big difference.
The hierarchy's depth is limited to $O(|V| + |C|)$ levels until all variables are
assigned.

D. Schreiber <dominik.schreiber@kit.edu> 2023
