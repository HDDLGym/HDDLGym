import numpy as np
import torch
import copy

from hddl_utils import convert_grounded_to_lifted_index, enumerate_index_to_binary_matrix
from learning_methods_lifted import get_grounded_prob_list_from_policy_output, enumerate_action, enumerate_state
from central_planner import centralized_planner


def run_planner_and_get_action_dict(policy_list, env, opt, deterministic=False, debug=False):
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
    if opt.use_central_planner:
        # 1. If plan with centralized planner:
        env.agents = centralized_planner(env, all_agents = env.agents, all_policies = policy_list,\
          debug=debug, deterministic=deterministic, device=opt.dvc, time_limit=opt.planner_time_limit)
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
    - logprob_a: a float number
    '''
    logprob_a = 0
    prob_list = policy.select_action(s_num,value=True)
    for agent in env.agents:
        if len(agent.prob_hierarchy)>0:
            # logprob_a += np.sum(np.log(np.array(agent.prob_hierarchy)))/len(agent.prob_hierarchy)
            prob_oper = get_grounded_prob_list_from_policy_output(agent.task_method_hierarchy, prob_list, env.env_dictionary,device=False)
            logprob_a += torch.mean(prob_oper)
            # logprob_a += torch.sum(torch.log(torch.tensor(agent.prob_hierarchy, device=opt.dvc)))/len(agent.prob_hierarchy)
    logprob_a = logprob_a/len(env.agents)
    logprob_a = logprob_a.clone().detach()
    return logprob_a
      


