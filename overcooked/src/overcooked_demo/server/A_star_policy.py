from overcooked_ai_py.mdp.actions import Action, Direction
import numpy as np
import networkx as nx


def navigate_to_target(lossless_state, start, goals, orientation, state, avoid_player=True, avoid_player_number=0):
    """
    Directs the agent towards a target location while optionally avoiding another player.
    """    
    avoid_player = False
    player1 = state.players[0]
    player2 = state.players[1]
    obstacle_map = np.sum(lossless_state[0][:, :, 10:16], axis=2)
    grid = obstacle_map

    if avoid_player and avoid_player_number == 0:
        obstacle_map[player1.position] = 1

    if avoid_player and avoid_player_number == 1:
        obstacle_map[player2.position] = 1

    if not isinstance(goals,list):
        goals = [goals]

    goal = goals[0]


    goal = None
    for g in goals:
        grid_copy = grid.copy()
        len_done = 2 if grid_copy[g[0], g[1]] == 1 else 1
        grid_copy[g[0], g[1]] = 0
        path, is_path = find_closest_reachable_node(grid_copy, start, g)
        if is_path and path[-1] == g:
            goal = g
            break

    if goal is None:
        return Action.STAY, False

    path_length = len(path)

    done = False
    if path_length == len_done:
        updated_position = (start[0] + orientation[0], start[1] + orientation[1])
        done = updated_position == goal

    if path_length == 0:
        return Action.STAY, done 

    directions = get_directions_from_path(path)
    ACTION = get_direction(directions)
    return ACTION, done


def get_direction(directions):
    if not directions:
        return Action.STAY
    direction_map = {"RIGHT": Direction.EAST, "LEFT": Direction.WEST, "UP": Direction.NORTH, "DOWN": Direction.SOUTH}
    return direction_map.get(directions[0], Action.STAY)

def build_graph_from_grid(grid):
    G = nx.grid_2d_graph(grid.shape[0], grid.shape[1])
    G.remove_nodes_from([(y, x) for y in range(grid.shape[0]) for x in range(grid.shape[1]) if grid[y, x] == 1])
    return G

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def find_closest_reachable_node(grid, start, goal):
    G = build_graph_from_grid(grid)
    
    if start not in G:
        raise ValueError(f"Start node {start} is not in the graph.")
    if goal not in G:
        goal = find_valid_goal(G, start, goal)
        if goal is None:
            raise ValueError("No reachable goal node found.")
    
    try:
        path = nx.astar_path(G, start, goal, heuristic=heuristic)
        return path, True
    except (nx.NodeNotFound, nx.NetworkXNoPath):
        closest_node = min(
            (node for node in G.nodes if nx.has_path(G, start, node)),
            key=lambda node: heuristic(node, goal),
            default=None
        )
        return closest_node, False

def find_valid_goal(G, start, original_goal):
    closest_node = min(
        (node for node in G.nodes if nx.has_path(G, start, node)),
        key=lambda node: heuristic(node, original_goal),
        default=None
    )
    return closest_node

def get_directions_from_path(path):
    directions = []
    for i in range(1, len(path)):
        prev, curr = path[i - 1], path[i]
        if curr == (prev[0], prev[1] + 1):
            directions.append("DOWN")
        elif curr == (prev[0], prev[1] - 1):
            directions.append("UP")
        elif curr == (prev[0] + 1, prev[1]):
            directions.append("RIGHT")
        elif curr == (prev[0] - 1, prev[1]):
            directions.append("LEFT")
    return directions



