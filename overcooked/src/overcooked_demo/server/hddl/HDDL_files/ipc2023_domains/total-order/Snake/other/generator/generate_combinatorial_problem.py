
import sys
import random

"""
Generate a combinatorial Snake problem.
The problem features k slots of size 2xd where d=2,3,4,...,k-1.
Each slot has a single point of entry and two mice at the back.
The snake therefore has to enter each slot and get out of each
but the final slot. Since the snake can only leave a slot of depth d
if its length is at most 2*d-2, there is only a single valid order
in which the snake can visit the slots.
Example output with three slots:

@          
# ## ## ##
#  #  #  #
#  #  #**#
#**#  ####
####**#
   ####
  3  4  2
 ≤4 ≤6 ≤2
"""

if len(sys.argv) <= 1:
    print(f"Usage: {sys.argv[0]} <nslots> [<seed>]")
    exit(0)

nslots = int(sys.argv[1])
seed = 1
if len(sys.argv) > 2:
    seed = int(sys.argv[2])
random.seed(seed)

# width of the grid
w = 1 + 3*nslots

# first row - snake of lenth 2, nothing else
print("$@" + ((w-2)*" "))

slotsizes = [2*i+2 for i in range(nslots)]
random.shuffle(slotsizes)
#print(slotsizes)

print("#" + (nslots*" ##"))
depth = 2
while True:
    row = "#"
    for slotsize in slotsizes:
        if slotsize+2 == depth:
            row += "**#"
        elif slotsize+2 > depth:
            row += "  #"
        else:
            row += "###"
    print(row)
    depth += 2
    if " " not in row and "*" not in row:
        break
