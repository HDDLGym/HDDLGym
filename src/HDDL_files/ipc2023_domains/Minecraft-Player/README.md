## Domain's Author
Julia Wichlacz <wichlacz@cs.uni-saarland.de>

## Domain's Paper
not published yet

## Modifications for HDDLGym:
We modified the original domain and problem files to adapt to HDDLGym system (the modified files are saved with '_hddlgym' in their file names):

*Domain file (saved as domain_hddlgym.hddl):*
- Add effects to the tasks that are trivial with existing predicates, which are tasks placeblockabstract, removeblockabstract, and findway.

- The other tasks (build...) are not trivial to include effects with existing predicates, therefore, we add new predicates and actions to help marking the completion of the tasks.

- Add 'agent' to types, actions, methods that include actions as subtasks

- Add the 'none' action.

- Add player1 as an agent type constant, so do not need to modify problem file

*Problem files (safed with ..._hddlgym.hddl):*

- Users can add more players to the object list of the problem if want to make it a multi-agent problem.