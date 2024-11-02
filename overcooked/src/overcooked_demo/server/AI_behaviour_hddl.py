from overcooked_ai_py.mdp.actions import Action, Direction
import numpy as np
from collections import deque
from overcooked_ai_py.mdp.overcooked_mdp import Recipe
import sys
import os
import random
from A_star_policy import navigate_to_target
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir + '/hddl')
from main_train import opt
from run_hddl_policy import run_hddl_model


class ExperimentHDDL:
    """Class for handling agent actions in the HDDL environment."""

    def __init__(self, player_idx):
       
        self.curr_phase = -1
        self.curr_tick = -1
        self.change = False
        self.action_complete = True
        self.player_idx = player_idx
        self.current_hddl_actions = None
                     

    def action(self, state, game_state, lossless_state):
            self.curr_tick += 1

            if not game_state.has_created_action_list:
                game_state.has_created_action_list = True
                game_state.action_list, game_state.hierarchies_array_list, _, _ = run_hddl_model(opt)

            game_state.current_hddl_actions = game_state.action_list[game_state.ActionCount]

            if game_state.IsActionCompleted_Agent1 and game_state.IsActionCompleted_Agent2:
                game_state.ActionCount += 1
                if game_state.ActionCount >= len(game_state.action_list): game_state.ActionCount = 0
                game_state.current_hddl_actions = game_state.action_list[game_state.ActionCount]
                game_state.IsActionCompleted_Agent1 = False         
                game_state.IsActionCompleted_Agent2 = False         


            if self.player_idx == 0:
                ACTION = self._process_agent_action(game_state, state, lossless_state, 'chef1', 1)
            elif self.player_idx == 1:
                ACTION = self._process_agent_action(game_state, state, lossless_state, 'chef2', 2)

            # print(game_state.hierarchies_array_list[game_state.ActionCount])

            return ACTION, None

    def reset(self):
        self.curr_tick = -1
        self.curr_phase += 1
        self.change = True


    def _process_agent_action(self, game, state, lossless_state, action_key, agent_id):

        current_action = game.current_hddl_actions[action_key]
        action_elements = current_action.split()

        if 'none' in action_elements:
            ACTION, IsBlocking = none_action_primitive(state, game, lossless_state, True, *action_elements)
            if IsBlocking: return ACTION

        is_action_completed = f'IsActionCompleted_Agent{agent_id}'
        if getattr(game, is_action_completed):
            return Action.STAY

        # Map primitive HDDL acitons to corresponding functions
        action_map = {
            'none': lambda: (Action.STAY, True),
            'wait': lambda: wait_action_primitive(state, game, lossless_state, True, *action_elements),
            'a-interact': lambda: interact_action_primitive(state, game, lossless_state, True, *action_elements)
        }

        for action_type, action_func in action_map.items():
            if action_type in action_elements:
                ACTION, completed = action_func()
                setattr(game, is_action_completed, completed)
                return ACTION
            
        return Action.STAY



# ===========================================================================================================
# ACTION PRIMITIVES

def none_action_primitive(state, game_state, lossless_state, avoid_player, *current_action_elements):
    _, player_label = current_action_elements
    player_number = int(player_label[-1:]) - 1
    other_player_number = (player_number - 1)*-1

    player = state.players[player_number]
    other_player = state.players[other_player_number]

    new_pos = tuple(p + o for p, o in zip(other_player.position, other_player.orientation))
    IsBlocking = (new_pos == player.position)

    player_pos_north = tuple(p + o for p, o in    zip(player.position, Direction.NORTH))
    player_pos_south = tuple(p + o for p, o in  zip(player.position, Direction.SOUTH))
    player_pos_east = tuple(p + o for p, o in  zip(player.position, Direction.EAST))
    player_pos_west = tuple(p + o for p, o in zip(player.position, Direction.WEST))

    obstacle_map = np.sum(lossless_state[0][:, :, 10:16], axis=2)
    obstacle_map[other_player.position] = 1

    north_available_bool = obstacle_map[player_pos_north] == 0
    south_available_bool= obstacle_map[player_pos_south] == 0
    east_available_bool = obstacle_map[player_pos_east] == 0
    west_available_bool= obstacle_map[player_pos_west] == 0

    available_directions_bool = [north_available_bool, south_available_bool, east_available_bool, west_available_bool]

    if IsBlocking:
        directions = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
        available_directions = [direction for direction, keep in zip(directions, available_directions_bool) if keep == 1]
        random_direction = random.choice(available_directions)
        return random_direction, True

    return Action.STAY, False


def interact_action_primitive(state, game_state, lossless_state, avoid_player, *current_action_elements):

    _, player_label, target_location, *_ = current_action_elements

    ACTION = Action.STAY

    if player_label == 'chef1' or player_label == 'chef2':
        player_number = int(player_label[-1:]) - 1
        if target_location == 'onion-pile':
            object = 'onion'
            object_number = 0
            ACTION, action_complete = execute_object_interaction(state, game_state, lossless_state, avoid_player, player_number, object, object_number, "pick")

        if target_location == 'bowl-pile':
            object = 'dish'
            object_number = 0

            player = state.players[player_number]
            put_down_object_holding = False
            if player.has_object():
                object_name = player.get_object().name
                if object_name != object:
                    put_down_object_holding = True

            if put_down_object_holding:
                object = 'counter'
                object_number = 11
                ACTION, _ = execute_object_interaction(state, game_state, lossless_state, avoid_player, player_number, object, object_number, "drop")           
                action_complete = False
            else:
                ACTION, action_complete = execute_object_interaction(state, game_state, lossless_state, avoid_player, player_number, object, object_number, "pick")

        if target_location == 'pot1':
            object = 'pot'
            object_number = 0

            pot_pos = game_state.mdp.get_pot_locations()[0]
            soup_exists = state.has_object(pot_pos)

            if IsChefHoldingSoup(state, player_number):
                return Action.STAY, True

            if not soup_exists:
                ACTION, action_complete = execute_object_interaction(state, game_state, lossless_state, avoid_player, player_number, object, object_number, "drop")
            
            if soup_exists:
                soup = state.get_object(pot_pos)
                if soup.is_ready:
                    ACTION, action_complete = pick_up_soup(state, game_state, lossless_state, object_number=object_number, avoid_player=True, player_number=player_number)
                else:
                    ACTION, action_complete = start_timer(state, game_state, lossless_state, object_number=object_number, avoid_player=avoid_player, player_number=player_number)

        if target_location == 'delivery':
            object = 'serve'
            object_number = 0
            ACTION, action_complete = execute_object_interaction(state, game_state, lossless_state, avoid_player, player_number, object, object_number, "drop")

    return ACTION, action_complete


def wait_action_primitive(state, game_state, lossless_state, avoid_player, *current_action_elements):

    _, player_label, target_location, _, _, = current_action_elements

    ACTION = Action.STAY
    IsActionComplete = False

    if player_label == 'chef1' or player_label == 'chef2':

        if target_location == 'pot1':

            pot_pos = game_state.mdp.get_pot_locations()[0]
            soup_exists = state.has_object(pot_pos)

            if soup_exists:
                soup = state.get_object(pot_pos)

                if soup.is_ready:
                    IsActionComplete = True

    return ACTION, IsActionComplete


# ===========================================================================================================
# HELPER FUNCTIONS

def execute_object_interaction(state, game_state, lossless_state, avoid_player, player_number, object, object_number, action_type):
    ACTION = Action.STAY
    action_complete = False

    if player_number == 0: game_state_action_count = game_state.action_list_count_0
    else: game_state_action_count = game_state.action_list_count_1

    if game_state_action_count == 0:
        if action_type == "pick":
            ACTION, done = pick_up_object(state, game_state, lossless_state, object=object, object_number=object_number, avoid_player=avoid_player, player_number=player_number)
        elif action_type == "drop":
             ACTION, done  = put_down_object(state, game_state, lossless_state, object, object_number, avoid_player=True, player_number=player_number)

        if done: game_state_action_count = 1

    if game_state_action_count == 1: action_complete = True
    return ACTION, action_complete


def IsChefHoldingSoup(state, player_number):
    player = state.players[player_number]    
    holding_object = player.has_object() and player.get_object().name == "soup"
    if holding_object:
        return True
    return False


def pick_up_soup(state, game_state, lossless_state, object_number=0, avoid_player=True, player_number=0):
    player = state.players[player_number]
    player_number_avoid = (player_number - 1) * -1

    start, orientation = player.position, player.orientation
    holding_object = player.has_object() and player.get_object().name == "soup"

    if holding_object:
        return Action.STAY, True

    pot_pos = game_state.mdp.get_pot_locations()[object_number] # first pot
    soup_exists = state.has_object(pot_pos)
    if not soup_exists:
        return Action.STAY, True

    soup = state.get_object(pot_pos)
    soup_is_ready = soup.is_ready
    
    goal = pot_pos
    ACTION, pick_up = navigate_to_target(lossless_state, start, goal, orientation, state, avoid_player=avoid_player, avoid_player_number=player_number_avoid)
    if pick_up and soup_is_ready:

        return Action.INTERACT, False
    if pick_up and not soup_is_ready:
        ACTION = Action.STAY

    return ACTION, False



def start_timer(state, game_state, lossless_state, object_number=0, avoid_player=True, player_number=0):
    player = state.players[player_number]
    player_number_avoid = (player_number - 1) * -1

    start, orientation = player.position, player.orientation
    pot_pos = game_state.mdp.get_pot_locations()[object_number] # first pot

    soup = state.get_object(pot_pos)
    soup_is_ready_or_cooking_or_empty = soup.is_ready or soup.is_cooking or (soup.ingredients == 0)

    if soup_is_ready_or_cooking_or_empty:
        return Action.STAY, True

    goal = pot_pos
    ACTION, pick_up = navigate_to_target(lossless_state, start, goal, orientation, state, avoid_player, avoid_player_number=player_number_avoid)
    if pick_up:
        ACTION = Action.INTERACT

    return ACTION, False


def put_down_object(state, game_state, lossless_state, object="pot", object_number=0, avoid_player=True, player_number=0):
    player = state.players[player_number]

    player_number_avoid = (player_number - 1) * -1
    player = state.players[player_number]

    holding_object = player.has_object()
    if not holding_object:
        return Action.STAY, True

    dropping_bowl = False
    if player.held_object.name == "dish":
        object = "counter"
        dropping_bowl = True

    start, orientation = player.position, player.orientation

    if object == "pot":
        object_loc = game_state.mdp.get_pot_locations()[object_number]
        
    if object == "serve":
        object_loc = game_state.mdp.get_serving_locations()[0]

    if object == "counter":
        object_loc = game_state.mdp.get_counter_locations()[11]
        
        if dropping_bowl:
            empty_counter_locations = game_state.mdp.get_empty_counter_locations(state)
            obstacle_map = np.sum(lossless_state[0][:, :, 10:16], axis=2)
            accessible_counters_list = accessible_counters(obstacle_map)
            
            object_loc = get_first_accessible_counter_location(empty_counter_locations, accessible_counters_list) or accessible_counters_list[9]


    goal = object_loc
    ACTION, pick_up = navigate_to_target(lossless_state, start, goal, orientation, state, avoid_player=avoid_player, avoid_player_number=player_number_avoid)
    if pick_up:
        ACTION = Action.INTERACT

    return ACTION, False



def pick_up_object(state, game_state, lossless_state, object="onion", location="dispenser",object_number=0, avoid_player=True, player_number=0):

    player_number_avoid = (player_number - 1) * -1
    player = state.players[player_number]

    if object == "onion": object_name = Recipe.ONION
    if object == "tomato": object_name = Recipe.TOMATO
    if object == "dish": object_name = "dish"
    if object == "soup": object_name = "soup"

    if object == "soup" and location == "dispenser":    
        return Action.STAY, True

    holding_object = player.has_object() and player.get_object().name == object_name

    if holding_object:
        return Action.STAY, True

    if location == "dispenser":
        if object == "onion":
            object_disp_loc = game_state.mdp.get_onion_dispenser_locations()[object_number]
        if object == "tomato":
            object_disp_loc = game_state.mdp.get_tomato_dispenser_locations()[0]
        if object == "dish":
            object_disp_loc = game_state.mdp.get_dish_dispenser_locations()[0]


    if location == "counter":
        counter_object_locations = game_state.mdp.get_counter_objects_dict(state)
        object_disp_loc = []
        if object == "onion":
            object_disp_loc = counter_object_locations.get('onion', [])
        if object == "tomato":
            object_disp_loc = counter_object_locations.get('tomato', [])
        if object == "dish":
            object_disp_loc = counter_object_locations.get('dish', [])
        if object == "soup":
            object_disp_loc = counter_object_locations.get('soup', [])


        if object_disp_loc == []:
            ACTION = Action.STAY
            return ACTION, False


    start, orientation = player.position, player.orientation
    goal = object_disp_loc

    ACTION, pick_up = navigate_to_target(lossless_state, start, goal, orientation, state, avoid_player=avoid_player, avoid_player_number=player_number_avoid)
    if pick_up:
        ACTION = Action.INTERACT

    return ACTION, False

def get_first_accessible_counter_location(empty_counter_locations, accessible_counters):
    for loc in empty_counter_locations:
        if loc in accessible_counters:
            return loc
    return None

def accessible_counters(obstacle_map):
    rows = len(obstacle_map)
    cols = len(obstacle_map[0])
    visited = [[False] * cols for _ in range(rows)]
    accessible_positions = set()
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    def bfs(start_row, start_col):
        queue = deque([(start_row, start_col)])
        visited[start_row][start_col] = True

        while queue:
            row, col = queue.popleft()

            for dr, dc in directions:
                new_row, new_col = row + dr, col + dc

                if 0 <= new_row < rows and 0 <= new_col < cols and not visited[new_row][new_col]:
                    if obstacle_map[new_row][new_col] == 1:
                        accessible_positions.add((new_row, new_col))
                        visited[new_row][new_col] = True
                    elif obstacle_map[new_row][new_col] == 0:
                        queue.append((new_row, new_col))
                        visited[new_row][new_col] = True

    for row in range(rows):
        for col in range(cols):
            if obstacle_map[row][col] == 0 and not visited[row][col]:
                bfs(row, col)

    return list(accessible_positions)


