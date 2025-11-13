We developed a simple Search and Rescue domain to study HDDLGym.

There are two heterogeneous types of agents: drones and ambulances.

Each drone can move from location l1 to l2 if the two are directly connected via (connect l1 l2) or (drone-connect l1 l2). In the problem file, drones are allowed to fly diagonally, in addition to moving vertically and horizontally between grid locations. You can modify this rule if you wish to enable drones to fly farther. The primary role of drones is to search for victims by flying around and observing each location to determine whether a victim is present.

Ambulances, on the other hand, can only move to adjacent locations that are connected horizontally or vertically. Their main function is to rescue victims once they have been found. However, in some cases, ambulances may also assist in the search, despite their limited mobility.



