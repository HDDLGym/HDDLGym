from hddl_utils import extract_object_list, check_subtask_with_world_state
from central_planner_utils import check_task_method_action, check_collaborative_status
from central_planner import centralized_planner
from learning_methods import get_observation_one_hot_vector
import copy
import numpy as np
import gym
from gym.spaces import Discrete, Box

class Agent():
  """
  Agent class:
  - initialized with agent_name, agent_types
  - self.update_agent(world): update agent's task_method_hierarchy by removing the action after it is performed.
  - self.update_hierarchy_agent(): update agents'task_method_hierarchy (remove the completed methods, tasks in the hierarchy)
  - self.update_hierarchy_agent_general
  - self.decentralize_planner_agent
  """
  def __init__(self, agent_name, agent_types, htn_goal = None, policy=None, belief_flag=False):
    self.name = agent_name # a string of agent name, eg. 'truck-0'
    self.types = agent_types # list of string of agent types, eg. ['vehicle', 'locatable']
    self.task_method_hierarchy = []
    self.completed_tasks_agent = [] # list of tasks (including primitive action) that agent has completed for the method, remove them after done method, and replace by task
    # self.completed_method_task = [] # list of tuple of (method, completed_subtask)
    self.infeasible_task_method = [] # list of infeasible task or method in each state, need to reset after every step
    self.logprob = 0
    self.prob = 1
    self.prob_hierarchy = []
    self.belief_other_agents = []
    self.agent_policy = policy
    self.htn_goal = htn_goal
    self.prev_task_method_hierarchy = []
    self.prev_prob_hierarchy = []
    self.belief_flag = belief_flag

  def __str__(self):
    ans =  self.name + ' has task_method_hierarchy:\n'
    for hierarchy_element in self.task_method_hierarchy:
      if isinstance(hierarchy_element, str):
        ans += hierarchy_element
        ans += ' => '
      elif hierarchy_element == None:
        hierarchy_element += 'None '
    return ans

  def __eq__(self, other):
    if isinstance(other, Agent):
      return self.name == other.name
    else:
      return False

  def get_agent_observation(self, world_current_state, env_dict, debug=False):
    '''
    '''
    all_operators = []
    for agent_ in [self]+self.belief_other_agents:
      all_operators += agent.prev_task_method_hierarchy
      if agent.htn_goal != None:
        all_operators += agent.htn_goal.pending_tasks
        all_operators += agent.htn_goal.remaining_tasks
    all_operators = set(all_operators)
    all_operators = list(all_operators)
    observation_num = get_observation_one_hot_vector(world.current_state, all_operators, world.env_dictionary)
    
    return observation_num


  def update_agent_hierarchy_by_checking_with_world_state(self, current_state, env_dictionary, debug=False):
    ''' Update agent and its belief other agents' hierarchies
    inputs:
    - current_state: list of grounded predicates of current state of the world
    - env_dictionary: dictionary of the environment
    - debug: boolean
    ouput: (void) internally update agent class 
    '''
    # print("calling update_agent_hierarchy_by_checking_with_world_state")
    # print("before update: hierarchy of agent {} is \n {}".format(self.name,self.task_method_hierarchy))
    self.infeasible_task_method = []
    prev_hierarchy = copy.deepcopy(self.task_method_hierarchy)
    self.prev_task_method_hierarchy = copy.deepcopy(self.task_method_hierarchy)
    self.prev_prob_hierarchy = copy.deepcopy(self.prob_hierarchy)
    for agent in [self] + self.belief_other_agents:
      # print("agent.infeasible_task_method:",agent.infeasible_task_method)
      agent.infeasible_task_method = []
      smallest_completed_index = len(agent.task_method_hierarchy)
      for operator_index, operator in enumerate(agent.task_method_hierarchy):
        if check_task_method_action(env_dictionary, operator) in ['task','action']:
          completed_task_ = check_subtask_with_world_state(env_dictionary, current_state, [operator])
          if len(completed_task_) > 0:
            smallest_completed_index = operator_index
            agent.completed_tasks_agent = [operator]
            break
      agent.task_method_hierarchy = agent.task_method_hierarchy[:smallest_completed_index]
      agent.prob_hierarchy = agent.prob_hierarchy[:smallest_completed_index]
      agent.htn_goal.update_htn(env_dictionary, current_state)

    # print("after update: hierarchy of agent {} is \n {}".format(self.name,self.task_method_hierarchy))
    if prev_hierarchy != self.task_method_hierarchy and debug:
      print('** hierachy before general update of agent {} is {}'.format(self.name, prev_hierarchy))
      print('** hierachy after general udpate of agent {} is {}\n'.format(self.name, self.task_method_hierarchy))

    return
  
  def decentralize_planner_agent(self, world, belief_other_agents = [], agent_policy = None, other_agent_policy_list = [], debug=False, deterministic=False, device=False, time_limit=5):
    '''This function choose the next operators for the hierarchy, until it reachs action
    inputs:
    - world: mainly for world.state, world.env
    - belief_other_agents: list of instances of Agent with the deterministic belief in their hierarchy
    - agent_policy: policy of picking operators of the agent, default to None (random)
    - other_agent_policy (belief): list of policies of other agents (inferring/belief), if empty (default), assume to be similar to agent_policy
    - belief_policy: policy of predicting/inferring other agents' hierarchy based on their history of actions

    outputs: (void)
    - update the hierarchy of the agent up to action
    '''
    if not device:
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if len(belief_other_agents) == len(world.agents) - 1:
      self.belief_other_agents = belief_other_agents
    list_agents = [self] + self.belief_other_agents
    policy_list = [agent_policy] + other_agent_policy_list
    assert len(list_agents) == len(policy_list), "policy list (len of {}) doesn't match with number of agents {}".format(len(policy_list), len(list_agents))
    list_agents = centralized_planner(world, main_agent_index = 0,all_agents = list_agents, all_policies = policy_list, debug=debug,\
     deterministic=deterministic, device=device, time_limit = time_limit)
    self = list_agents[0]
    self.belief_other_agents = list_agents[1:]
    if debug:
      for belief_agent in self.belief_other_agents:
        print("Agent: belief {} has hierarchy after cenralized planner: {}".format(belief_agent.name, belief_agent.task_method_hierarchy))

    return



