import numpy as np
import torch
import copy
import time

# MODIFY if NEEDED: Change learning_methods to your defined learning methods file
from learning_methods import get_grounded_prob_list_from_policy_output

from hddl_utils import extract_object_list
from central_planner import centralized_planner

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

def plan_for_agent(i, ag, world_info, policy_list, dvc, planner_time_limit, deterministic):
    '''This function runs the planner for each agent to update their action hierarchy'''
    belief_other_agents = []  # comment this line if want manually embed belief to be groundtruth
    other_agent_policy_list = copy.deepcopy(policy_list)
    other_agent_policy_list.pop(i)
    # Run decentralized planner for the agent
    ag.decentralize_planner_agent(
        world_info, 
        belief_other_agents=belief_other_agents, 
        agent_policy=policy_list[i],
        other_agent_policy_list=other_agent_policy_list, 
        device=dvc,
        deterministic=deterministic, 
        time_limit=planner_time_limit
    )

def parallelly_run_planner_and_get_action_dict(policy_list, env, opt, deterministic=False, debug=False):
    '''This function runs the planner parallelly for each agent to update their action hierarchy 
    and generate action dictionary for the step function.
    inputs:
    - policy_list: list of policies of all agents
    - env: an instance of HDDLEnv
    - opt: parameters
    - debug: boolean, whether to run the planner in a debug mode
    - deterministic: boolean, whether to run the planner in a deterministic or probabilistic way
    outputs:
    - action_dict: dictionary of {agent_name: string of grounded action}
    - env: updated instance of HDDLEnv
    - hierarchies: a record of action hierarchies of all agents
    '''
    if opt.use_central_planner:
        env.agents = centralized_planner(env, all_agents=env.agents, all_policies=policy_list,
                                         debug=debug, deterministic=deterministic, device=opt.dvc, time_limit=opt.planner_time_limit)
    else:
        # Run decentralized plan for each agent in parallel
        # Create a copy of short version of the environment for each agent (world_info)
        world_info_list = [copy.deepcopy(env.extract_world_info())] * env.num_agents
        policy_list_cpu = copy.deepcopy(policy_list)
        # for policy_dict in policy_list_cpu:
        #     for key, value in policy_dict.items():
        #         value.dvc = 'cpu'
        #         policy_dict[key] = value.to('cpu')
        policy_list_all = [policy_list_cpu] * env.num_agents

        # Use ProcessPoolExecutor to run each agent's planning in parallel
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(plan_for_agent, i, agent, world_info_list[i], policy_list_all[i],\
                                 opt.dvc, opt.planner_time_limit, deterministic) 
                for i, agent in enumerate(env.agents)
            ]
            for future in futures:
                future.result()  # Wait for all threads to complete
            
        
    # Extract action and convert to dict of string
    action_dict = {}
    hierarchies = []
    for agent_for_action in env.agents:
        hierarchies.append(agent_for_action.task_method_hierarchy)
        if len(agent_for_action.task_method_hierarchy) > 0:
            action_dict[agent_for_action.name] = agent_for_action.task_method_hierarchy[-1]
        else:
            action_dict[agent_for_action.name] = 'none {}'.format(agent_for_action.name)
            print("assign action none to {} bc no hierarchy".format(agent_for_action.name))
    return action_dict, env, hierarchies

def check_task_method_action(env_dictionary, operator_str):
  '''check if string s is task, method, or action
  return 'task' or 'method' or 'action'
  inputs:
  - env_dictionary: dictionary of environment
  - s: string of operator
  '''
  if operator_str == None:
    return None
  is_task = False
  is_method = False
  is_action = False
  operator_name, operator_object = extract_object_list(operator_str)

  for lifted_task in env_dictionary['lifted tasks list']:
    if lifted_task.name == operator_name:
      is_task = True
      return 'task'

  if not is_task:
    for lifted_method in env_dictionary["lifted methods list"]:
      if lifted_method.name == operator_name:
        is_method = True
        return 'method'

  if not is_task and not is_method:
    for lifted_action in env_dictionary['lifted actions list']:
      if lifted_action.name == operator_name:
        is_action = True
        return 'action'

  assert is_task or is_method or is_action, "{} is neither task, method, nor action!".format(operator_str)
def run_planner_and_get_action_dict(policy_list, env, opt, deterministic=False, debug=False, compare_central_decentral = False):
    ''' This function run the planner to update action hierarchy of each agent and 
        get the action dictionary for the step function
        
        Inputs:
        - policy_list: list of policies of all agents
        - env: an instance of HDDLEnv
        - opt: parameters
        - deterministic: boolean, whether to run the planner in a deterministic or probabilistic way
        Output:
        - action_dict: dictionary of {agent_name: string of grounded action}
        - env: updated instance of HDDLEnv
        - hierarchies: a record of action hierarchies of all agents
    '''
    # Run the planner:
    start_time = time.time()
    world_info = env.extract_world_info()
    if opt.use_central_planner:
        # 1. If plan with centralized planner:
        env.agents = centralized_planner(world_info, all_agents = env.agents, all_policies = policy_list,\
          debug=debug, deterministic=deterministic, device=opt.dvc, time_limit=opt.planner_time_limit)
    
    elif compare_central_decentral:
        # 1. Plan with centralized planner:
        _ = centralized_planner(world_info, all_agents = env.agents, all_policies = policy_list,\
            debug=debug, deterministic=deterministic, device=opt.dvc, time_limit=opt.planner_time_limit)
        print("Centralized planner time: ", time.time()-start_time)
        start_time = time.time()
        # 2. run decentralized plan for each agent to get action for each agent at each step
        for i,ag in enumerate(copy.deepcopy(env.agents)):
            # belief_other_agents = copy.deepcopy(env.agents)
            # belief_other_agents.remove(ag)
            belief_other_agents = [] # comment this line if want manually embed belief to be groundtruth
            other_agent_policy_list = copy.deepcopy(policy_list)
            other_agent_policy_list.pop(i)
            ag.decentralize_planner_agent(world_info, belief_other_agents = belief_other_agents,agent_policy = policy_list[i],\
              other_agent_policy_list = other_agent_policy_list, device=opt.dvc, deterministic=deterministic, \
              time_limit=opt.planner_time_limit)
        print("Decentralized planner time: ", time.time()-start_time)
    
    else:
        # 2. run decentralized plan for each agent to get action for each agent at each step
        for i,ag in enumerate(env.agents):
            # belief_other_agents = copy.deepcopy(env.agents)
            # belief_other_agents.remove(ag)
            belief_other_agents = [] # comment this line if want manually embed belief to be groundtruth
            other_agent_policy_list = copy.deepcopy(policy_list)
            other_agent_policy_list.pop(i)
            ag.decentralize_planner_agent(env, belief_other_agents = belief_other_agents,agent_policy = policy_list[i],\
              other_agent_policy_list = other_agent_policy_list, device=opt.dvc, deterministic=deterministic, \
              time_limit=opt.planner_time_limit)

    ############
    #extract action and convert to dict of string
    action_dict = {}
    hierarchies = []
    for agent_for_action in env.agents:
        hierarchies.append(agent_for_action.task_method_hierarchy)
        if len(agent_for_action.task_method_hierarchy)>0:
            action_dict[agent_for_action.name] = agent_for_action.task_method_hierarchy[-1]
        else:
            action_dict[agent_for_action.name] = 'none {}'.format(agent_for_action.name)
            print("assign action none to {} bc no hierarchy".format(agent_for_action.name))
    return action_dict, env, hierarchies



def get_logprob(s_num, policy, env, opt):
    '''Get log of probability of actions of all agents in the env
    inputs:
    - s_num: state in one-hot array
    - policy: RL policy
    - env: an instance of HDDLEnv 
    - opt: parameters
    output:
    - logprob_a: a float number, it is the average of log probability of all operators of all agents
    '''
    logprob_a = 0
    prob_list = policy.select_action(s_num,value=True)
    for agent in env.agents:
        if len(agent.prob_hierarchy)>0:
            # logprob_a += np.sum(np.log(np.array(agent.prob_hierarchy)))/len(agent.prob_hierarchy)
            prob_oper = get_grounded_prob_list_from_policy_output(agent.task_method_hierarchy, prob_list, env.env_dictionary,device=opt.dvc)
            logprob_a += torch.mean(torch.log(prob_oper))
            # logprob_a += torch.sum(torch.log(torch.tensor(agent.prob_hierarchy, device=opt.dvc)))/len(agent.prob_hierarchy)
    logprob_a = logprob_a/len(env.agents)
    logprob_a = logprob_a.clone().detach()
    return logprob_a
      


