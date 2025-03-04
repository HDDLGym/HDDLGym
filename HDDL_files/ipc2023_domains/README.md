# ipc2023_domains

## Licenses
All the domains here are copied from the IPC 2023: https://github.com/ipc2023-htn/ipc2023-domains

Each domain include a new HDDL domain file that is specially modified on top of the original HDDL domain file to run with HDDLgym

Notes for HDDL files to be used with HDDLGym:
- Please remove all comments (lines start with ';') from the HDDL files.
- Names of any parameter should NOT be a subset of any other parameters' name in the same tasks, methods, or actions. For example:
  + Instead of -- *:parameters ?agent ?agent-1 - agent*,
  + Do -- *:parameters ?agent-0 ?agent-1 - agent*
