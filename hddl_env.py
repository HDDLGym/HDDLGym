from hddl_utils import extract_object_list, extract_object_list, Action
from hddl_parser import main_parser
from central_planner_utils import check_task_method_action, check_collaborative_status, check_valid_combination_operator
from learning_methods_lifted import get_observation_space, get_action_space
import copy
import random
import agent_class
import gymnasium as gym
import itertools

class HDDLEnv(gym.Env):
    def __init__(self, domain_file, problem_file, policy_list = []):
        self.env_dictionary = main_parser(domain_file, problem_file)
        self.current_state = self.env_dictionary['initial state']
        self.step_count = 0
        self.agents = []
        self.done = False
        # # observation is list of grounded dynamic predicates + lifted operators + objects
        # self.observation_space = gym.spaces.Discrete(n=len(self.env_dictionary['grounded dynamic predicate list'])+\
        #   len(self.env_dictionary['lifted operators list'])+len(self.env_dictionary['objects list']))
        # # action is list of lifted operators + objects
        # self.action_space = gym.spaces.Discrete(n=len(self.env_dictionary['lifted operators list'])+len(self.env_dictionary['objects list']))
        self.observation_space = get_observation_space(self.env_dictionary)
        self.action_space = get_action_space(self.env_dictionary)
        self._max_episode_steps = 25
        self.reset(policy_list = policy_list)
        self.history_action_dict = []

    def reset(self, policy_list = []):
        ''' reset the env to the initial state, including:
          1. reset current state to initial_state
          2. step count to 0
          3. done <-- False
          4. reset HTN goal to initial state
          5. initialize agents
        '''
        self.current_state = self.env_dictionary['initial state']
        self.step_count = 0
        self.done = False
        self.history_action_dict = []
        self.env_dictionary['htn goal'].reset()
        htn_goal = copy.deepcopy(self.env_dictionary['htn goal'])
        self.initialize_agents(policy_list = policy_list, htn_goal=htn_goal)

    def initialize_agents(self, policy_list, htn_goal=None):
      '''Initiate agents' instances, using information from self.env_dictionary
      - inputs:
        - policy_list: list of policies
        - htn_goal: the overall goal, if None, set it to initial htn_goal mentioned in the problem.hddl
      - output: void
        - update self.agents, belief_other_agents of each agent
      '''
      self.agents = [] #reset the list to empty
      for i, agent_name in enumerate(self.env_dictionary['type object dict']['agent']):
        #a should be name of the agent
        a_types = []
        for type_ in self.env_dictionary['type object dict'].keys():
          if agent_name in self.env_dictionary['type object dict'][type_]:
            a_types.append(type_)
        if len(policy_list) == len(self.env_dictionary['type object dict']['agent']):
          policy = policy_list[i]
        else:# set policy to None
          policy = None
        if htn_goal == None: # set to default htn goal from the env
          htn_goal = copy.deepcopy(self.env_dictionary['htn goal'])
        agent_ = agent_class.Agent(agent_name, a_types, htn_goal=htn_goal, policy= policy)
        self.agents.append(agent_)
      for agent in self.agents:
        other_agents = copy.deepcopy(self.agents)
        other_agents.remove(agent)
        agent.belief_other_agents = other_agents
        # set policy of each belief other agent be the policy of the agent:
        for belief_agent in agent.belief_other_agents:
          belief_agent.agent_policy = agent.agent_policy


    
    def step(self, action, debug=False):
        ''' step thru env with action:
        inputs:
        - action: dictionary {agent_name: action_string}
        - debug: boolean, default to False
        output: 
        - new_state: list of grounded predicates of new world state
        - reward: reward (float)
        - goals_reached: boolean if all goals are completed
        - truncated: boolean, True when the step number exceeds max_episode_steps
        - debug_info: list of completed goal tasks
        '''
        self.history_action_dict.append(action)
        self.step_count += 1
        # Check conflict:
        action_list = list(action.values())
        no_conflict = True
        if len(action_list) >= 2: #only check conflict for multi-agent contexts
          no_conflict = check_valid_combination_operator(self.current_state, self.env_dictionary, action_list, self.agents, debug=debug)
        new_states = []
        costs = []
        new_state = copy.deepcopy(self.current_state)
        if no_conflict:
          for agent_name in action.keys():
              new_state, cost = self.apply_action(action[agent_name], new_state)
              new_states.append(new_state)
              costs.append(cost)
        # Apply any environment actions that are applicable to the state:
        valid_grounded_env_action_list = self.valid_grounded_env_action(new_state, debug=False)
        for environment_action in valid_grounded_env_action_list:
            new_state, cost = self.apply_env_action(environment_action, new_state, debug=False)
        self.prev_state = copy.deepcopy(self.current_state)
        self.current_state = new_state
        if debug:
            print("current state after apply action to all agents:", self.current_state)
        #update and check goals:
        goals_reached = self.update_and_check_goal()
        truncated = False
        if self.step_count >= self._max_episode_steps:
            # print("truncated bc step >= max_ep_steps")
            truncated = True
        reward = self.get_reward(costs, goals_reached, no_conflict)
        self.done = goals_reached or truncated
        debug_info = self.env_dictionary['htn goal'].completed_tasks
        return new_state, reward, goals_reached, truncated, debug_info

    def apply_action(self, action_str, state, debug=False):
        '''Apply action to the world, return new_state and cost
        inputs:
        - action_str: string of grounded action
        - state: list of predicates (list of string) that represents the current state of the world
        - debug: boolean
        outputs:
        - new_state: list of predicates of the updated state
        - cost: cost of the action
        '''
        # if 'none' in action_str:
        #   debug=True
        new_state = copy.deepcopy(state)
        if debug:
          print('Apply action {} to the state {}'.format(action_str, '\n'.join(state)))
        action_name, action_objects = extract_object_list(action_str)
        matched_action = None
        for action in self.env_dictionary['lifted actions list']:
          if action.name == action_name:
            matched_action = action
            break
        assert isinstance(matched_action, Action), "cannot find action {} in the list".format(action_str)
        var_obj_dict = matched_action.translate_objects_list_to_dict(action_objects)
        new_state, cost = matched_action.apply_effect(state, var_obj_dict)
        if debug:
          print("after action {} is applied, new state is: {}".format(action_str, '\n'.join(new_state)))

        return new_state, cost

    def apply_env_action(self, env_action_str, state, debug=False):
        '''
        Apply environment action to the state if the precondition is satisfied
        input:
        - env_action_str: string of grounded environment action
        - state: list of predicates
        - debug: boolean, indicate debug mode
        output:
        - new_state: updated list of predicates
        '''
        new_state = copy.deepcopy(state)
        # 1. Find matching lifted env action:
        action_name, action_objects = extract_object_list(env_action_str)
        matched_action = None
        for action in self.env_dictionary['lifted actions list']:
          if action.name == action_name:
            matched_action = action
            break
        assert isinstance(matched_action, Action), "cannot find action {} in the list".format(env_action_str)
        var_obj_dict = matched_action.translate_objects_list_to_dict(action_objects)

        # check the precondition:
        valid_precondition = matched_action.check_precondition(state, var_obj_dict, debug=False)
        
        # execute this action if the precondition is correct:
        if valid_precondition: # extra cautious
          new_state, cost = matched_action.apply_effect(state, var_obj_dict)
          if debug: 
            print("new state after apply action {}:\n  {}".format(env_action_str, '\n '.join(new_state)))
        else:
          if debug:
            print("HDDLEnv: the env_action {} is no longer valid".format(env_action_str))
          cost = 0
        
        return new_state, 0 #assume all env actions have cost 0
    
    def valid_grounded_env_action(self, state, debug=False):
        '''Return list of valid grounded environment action by checking precondition with the state.
        inputs:
        - state: list of predicates
        - debug: boolean
        outputs:
        - valid_grounded_env_action_list: list of string of grounded action
        '''
        valid_grounded_env_action_list = []
        for lifted_env_action in self.env_dictionary['lifted environment actions list']:
          var_options_list = list(self.env_dictionary['type object dict'][t] for t in lifted_env_action.parameters_type_list)
          var_combination_list = list(itertools.product(*var_options_list))
          for l in var_combination_list:
            var_object_dict = lifted_env_action.translate_objects_list_to_dict(l)
            if lifted_env_action.check_precondition(state,var_object_dict, debug=False):
              valid_grounded_env_action_list.append(lifted_env_action.print_name(object_list = l))
        
        if debug and len(valid_grounded_env_action_list) > 0:
          print("list of valid grounded env action:", valid_grounded_env_action_list)
        return valid_grounded_env_action_list

    def update_and_check_goal(self):
        '''check if the current state meet goals:
        inputs: 
        - completed_tasks: list of completed tasks of agents
        output:
        - done: boolean: if all htn goal tasks are completed
        '''
        self.env_dictionary['htn goal'].update_htn(self.env_dictionary, self.current_state)
        done = len(self.env_dictionary['htn goal'].tasks) == len(self.env_dictionary['htn goal'].completed_tasks)

        return done

    def get_reward(self, costs, goals_reached, no_conflict):
        '''
        '''
        reward = 0
        reward -= 0# sum(costs)
        if not no_conflict:
            # if conflict:
            reward -= 0#10
        if goals_reached:
            reward += 1#100
        return reward
    