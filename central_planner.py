from central_planner_utils import check_task_method_action, choose_method_policy, choose_subtask_policy
from central_planner_utils import generate_valid_operators, check_valid_combination_operator
from hddl_utils import extract_object_list, enumerate_list, convert_grounded_to_lifted_index
from hddl_utils import enumerate_index_to_binary_matrix
from hddl_utils import find_corresponding_task
from learning_methods_lifted import get_grounded_prob_list_from_policy_output, get_observation_one_hot_vector, get_probabilities_of_operators
from learning_methods_lifted import get_probability_of_valid_operator_combinations
import numpy as np
import random
import itertools
import copy
import torch
import time


def centralized_planner(world, main_agent_index=0, all_agents = [], all_policies = [], debug=False, deterministic=False, device = False, time_limit = 60):
  ''' This function plan the next operators of all agents up until all of them reach actions
  inputs:
  - world: using info of world.current_state, world.env_ditionary dict
  - all_agents: list of instances of Agent, if empty (default), set it to copy.deepcopy(world.agents)
  - all_policies: list of policies of all agents, if empty (default), use random policy for all of them
  - debug: boolean
  - deterministic: boolean, indicate whether choose the operators based on their probability or have some randomness with the prob
  - device: for torch device
  - time_limit: int, end the planner for this step if spending more than time_limit seconds, the hierarchy of each agent be ['none agent-name']
  outputs:
  - list of all Agent instances that have updated hierarchies

  Approach:
  1. Find a valid list of operator of the agent, giving the current hierarchy, use policy to find prob list of them
  2. Create combination list of all possible comb from the lists of opers, and validate each combination
  3. Multiply the prob list of valid combination to get a list of comb_prob of all combination
  4. Pick the comb by random.choice with weights are com_prob, or argmax from prob_list if deterministic is True
  5. Update the hierarchy of the agents: for each agent:
    5.1. if agent is mentioned in only 1 oper (or 2 with 1 is 'none'), update the agent's hierarchy with it
    5.2. if agent appears in more than 2 opers: use agent's policy to choose (random.choices with weight is prob from the policy)
    5.3. if agent has already reached action, but after 5.1 and 5.2, the chosen oper is different from its actions, replace the last element of hierarchy
    5.4. update the agent_reach_action list according to the current hierarchies
  6. Repeat 1-5 untill all agents reach actions
  '''
  # debug=True
  # 0. Common parameters:
  if not device:
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  infeasible_com_list = list()
  if len(all_agents) != len(world.agents):
    agents = world.agents
  else:
    agents = all_agents

  if len(all_policies) == 0:
    all_policies = [None] * len(world.agents)
  main_policy = all_policies[main_agent_index]
  state = world.current_state
  
  # List of all operators for observation
  all_operators = []
  for agent in agents:
    all_operators += agent.prev_task_method_hierarchy
    if agent.htn_goal != None:
      all_operators += agent.htn_goal.pending_tasks
      all_operators += agent.htn_goal.remaining_tasks
  all_operators = set(all_operators)
  all_operators = list(all_operators)
  observation_num = get_observation_one_hot_vector(world.current_state, all_operators, world.env_dictionary, device = device)
  agent_reach_action = [] # list of name of agents who have reach their action in their task_method_hierarchy
  if debug: print("---Start while loop...")
  time_start = time.time()
  while len(agent_reach_action) < len(agents):
    elapsed_time = time.time() - time_start
    if elapsed_time > time_limit:
      print("Centralized Planner's running time has reach the time limit of {} seconds, setting incomplete agents' hierarchy to ['none']!".format(time_limit))
      for agent in agents:
        if agent.name not in agent_reach_action:
          agent.task_method_hierarchy = ['none '+agent.name]
          agent.prob_hierarchy = [1]
      break # break the while loop
    valid_operators_dict = dict()
    # 1. Find a valid list of operator of the agent, giving the current hierarchy, use policy to find prob list of them
    current_hierachy = []
    for i_a1, agent in enumerate(agents):
      if len(agent.task_method_hierarchy) > 0:
        current_hierachy.append(agent.task_method_hierarchy[-1])
      if agent.name in agent_reach_action:
        valid_operators_dict[agent.name] = [agent.task_method_hierarchy[-1]]
        # print("CCP377: valid oper is copied for {} with last element of hierarchy {}".format(agent.name, agent.task_method_hierarchy[-1]))
      else: # agent.name not in agent_reach_action
        has_valid = False
        while not has_valid: #keep modifying the hierarchy until has valid following oper
          valid_op_list, empty_by_others = generate_valid_operators(world, agent, print_empty_by_others=True)
          if len(valid_op_list) == 0 and len(agent.task_method_hierarchy) > 0 and empty_by_others:
            # cannot move forward due to collab with others and they have not completed their tasks
            # append 'none' action to the valid_oper_list
            valid_operators_dict[agent.name] = ['none '+agent.name]
            has_valid = True
            if agent.name not in agent_reach_action:
              agent_reach_action.append(agent.name)
          elif len(valid_op_list) == 0 and len(agent.task_method_hierarchy) > 0 and not empty_by_others: 
            #No valid operator after task_method_hierarchy[-1]
            # print("No valid operator after ",agent.task_method_hierarchy[-1])
            agent.infeasible_task_method.append(agent.task_method_hierarchy[-1])
            # 6.26: if the infeasible task method is in other agents' hierarchy, remove them too:
            for other_agent_ in agents:
              if other_agent_.name != agent.name:
                if agent.task_method_hierarchy[-1] in other_agent_.task_method_hierarchy:
                  # print("CCP 389: need to remove infeasible oper {} from {} hierarchy {}".format(agent.task_method_hierarchy[-1], a_.name, a_.task_method_hierarchy))
                  infeasible_oper_index = other_agent_.task_method_hierarchy.index(agent.task_method_hierarchy[-1])
                  other_agent_.task_method_hierarchy = other_agent_.task_method_hierarchy[:infeasible_oper_index]
                  other_agent_.prob_hierarchy = other_agent_.prob_hierarchy[:infeasible_oper_index]
                  if other_agent_.name in agent_reach_action:
                    agent_reach_action.remove(other_agent_.name)
                  # print("CCP 398: Updated hierarchy of other ag {}: {}".format(a_.name, a_.task_method_hierarchy))
                  #update the valid_operators_dict: (6.27)
                  if len(other_agent_.task_method_hierarchy)>0:
                    valid_operators_dict[other_agent_.name] = [other_agent_.task_method_hierarchy[-1]]
                  else:
                    valid_operators_dict[other_agent_.name] = ['none ' + other_agent_.name]

            if agent.task_method_hierarchy[-1] in agents[main_agent_index].htn_goal.pending_tasks:
              agents[main_agent_index].htn_goal.remaining_tasks.append(agent.task_method_hierarchy[-1])
              agents[main_agent_index].htn_goal.pending_tasks.remove(agent.task_method_hierarchy[-1])
            agent.task_method_hierarchy.pop(-1)
            agent.prob_hierarchy.pop(-1)
          elif len(valid_op_list) != 0:
            # if there are actions in the valid op list, add 'none' to the list
            for op in valid_op_list:
              if check_task_method_action(world.env_dictionary, op) == 'action':
                valid_op_list = set(valid_op_list)
                valid_op_list.add('none '+agent.name)
                valid_op_list = list(valid_op_list)
                break
            # always add 'none' action to the valid_op_list:
            # valid_op_list.append('none '+agent.name)
            valid_operators_dict[agent.name] = valid_op_list
            has_valid = True
          elif len(agent.task_method_hierarchy) == 0 and len(valid_op_list) == 0: #no more valid operator
            valid_op_list = ['none {}'.format(agent.name)]
            # print("CCP389: no more valid operator so adding none to the valid list for {} with hierarchy \n{}\n".format(agent.name, '\n==>'.join(agent.task_method_hierarchy)))
            valid_operators_dict[agent.name] = valid_op_list
            has_valid=True

    # 2. Create combination list of all possible comb from the lists of opers, and validate each combination
    # print('valid_operators_dict:',valid_operators_dict)
    operator_options_list = list(valid_operators_dict[agent.name] for agent in agents)
    operator_combination_list = list(itertools.product(*operator_options_list))
    # print('operator_combination_list:', operator_combination_list)

    # 2.1. prune invalid combination:
    valid_operator_combination_list = []
    for com in operator_combination_list:
      if check_valid_combination_operator(world.current_state, world.env_dictionary, com, agents, infeasible_com_list = infeasible_com_list):
        valid_operator_combination_list.append(com)
    # Handle the case when there is nothing in valid_operator_combination_list:
    if len(valid_operator_combination_list) == 0:
      if debug:
        print('No valid combination amongst the list:',operator_combination_list)
      # remove the last layer of the hierarchies of agents and append them to the infeasible and remove any agents from reach_action_agents
      current_com = []
      for a__ in agents:
        if len(a__.task_method_hierarchy) == 0:
          current_com.append('')
        else:
          current_com.append(a__.task_method_hierarchy[-1])
          a__.task_method_hierarchy.pop(-1)
          a__.prob_hierarchy.pop(-1)
        if a__.name in agent_reach_action:
          agent_reach_action.remove(a__.name)
      infeasible_com_list.append(set(current_com))
      # continue the big while loop
      continue

    # 3. Multiply the prob list of valid combination to get a list of comb_prob of all combination
    # Use policy to get the probability list of all operators,
    #       then calculate the probability of each combination (product of each opertors)
    #       find do random choice with probability or just choose the max prob


    valid_prob_list = get_probability_of_valid_operator_combinations(valid_operator_combination_list, all_policies, main_agent_index, world.current_state, world.env_dictionary, agents, device=device)

    # 4. choose a combination by random choice with prob:
    if deterministic:
      chosen_combination = valid_operator_combination_list[torch.argmax(valid_prob_list).item()]
    else:
      # print("choose randomly with weights are prob")
      chosen_combination = random.choices(valid_operator_combination_list, weights = valid_prob_list, k=1)[0]
    
    if debug and len(valid_operator_combination_list) >=2 and 'none' in ' '.join(chosen_combination):
      print("\n***chosen com (CCP 465, centralized_planner) {} from list \n {} \n with prob {}".format(chosen_combination, valid_operator_combination_list,valid_prob_list))

    # 5. Update the hierarchy of the agents: for each agent:
    for index_agent, ag in enumerate(agents):
      factor = 1
      related_oper_indices = []
      update_oper = None
      for i_o, oper in enumerate(chosen_combination):
        if i_o == index_agent or ag.name in oper:
          #6.26: if the oper already appear in the hierarchy, not considering it
          if len(ag.task_method_hierarchy) <= 1:
            related_oper_indices.append(i_o)
          elif chosen_combination[i_o] not in ag.task_method_hierarchy[:-1] or chosen_combination[i_o] == 'none '+ag.name:
            related_oper_indices.append(i_o)
          else:
            # print('CCP 489: oper {} in the chosen comb {} appeared in the hierarhcy of {}: {}'.format(chosen_combination[i_o], chosen_combination, ag.name,ag.task_method_hierarchy))
            related_oper_indices.append(i_o) #TO DO: reconsider: should we do something for repeating the same task in the hierarchy

      # 5.1. if agent is mentioned in only 1 oper (or 2 with 1 is 'none'), update the agent's hierarchy with it
      if len(related_oper_indices) == 1:
        update_oper = chosen_combination[related_oper_indices[0]]
      elif (len(related_oper_indices) == 2 and 'none '+ag.name in chosen_combination and len(agents)<=2):
        #Prefer to choose operator other than none action. Note that this won't apply for 3+ agents
        not_none_index = int(1-chosen_combination.index('none '+ag.name))
        update_oper = chosen_combination[not_none_index]

      # 5.2. if agent appears in more than 2 opers: use agent's policy to choose (random.choices with weight is prob from the policy)
      elif len(related_oper_indices) >= 2:
        agent_policy = all_policies[index_agent]
        if agent_policy == None: #uniform probability when policy is None
          prob_op_list = [1/len(related_oper_indices)]*len(related_oper_indices)
        else:
          agent_prob_oper_list = agent_policy.select_action(observation_num, value=True)
          prob_op_list = []
          related_opers = [chosen_combination[com_i] for com_i in related_oper_indices]
          prob_op_list = get_grounded_prob_list_from_policy_output(related_opers, agent_prob_oper_list, world.env_dictionary, device=device)
          #normalize:
          prob_op_list = prob_op_list/(torch.sum(prob_op_list).item()+1e-20)
        if deterministic:
          update_oper_i = related_oper_indices[torch.argmax(prob_op_list).item()]
        else:
          update_oper_i = random.choices(related_oper_indices, weights = prob_op_list.tolist(), k=1)[0]
        update_oper = chosen_combination[update_oper_i]
        factor *= prob_op_list[related_oper_indices.index(update_oper_i)].item()

      # 5.3. if agent has already reached action, but after 5.1 and 5.2, the chosen oper is different from its actions, replace the last element of hierarchy
      if update_oper not in ag.task_method_hierarchy and ag.name in agent_reach_action:
        label_oper = check_task_method_action(world.env_dictionary, update_oper)
        if label_oper == 'action' and len(ag.task_method_hierarchy)>0:
          # print("pop out the current action {} before adding new action {}".format(ag.task_method_hierarchy),update_oper)
          ag.task_method_hierarchy.pop(-1)
          ag.prob_hierarchy.pop(-1)
          ag.task_method_hierarchy.append(update_oper)
          ag.prob_hierarchy.append(valid_prob_list[valid_operator_combination_list.index(chosen_combination)] * factor)
          # print("new hierarchy after replace action: ", ag.task_method_hierarchy)
        elif label_oper == 'task':
          agent_reach_action.remove(ag.name)
          ag.task_method_hierarchy.pop(-1)
          ag.prob_hierarchy.pop(-1)
          # print("Update hierarchy with new oper:", update_oper)
          ag.task_method_hierarchy.append(update_oper)
          ag.prob_hierarchy.append(valid_prob_list[valid_operator_combination_list.index(chosen_combination)] * factor)
        elif label_oper == 'method':
          agent_reach_action.remove(ag.name)
          ag.task_method_hierarchy.pop(-1)
          ag.prob_hierarchy.pop(-1)
          corresponding_task = find_corresponding_task(world.env_dictionary, update_oper)
          ag.task_method_hierarchy.append(corresponding_task)
          ag.prob_hierarchy.append(valid_prob_list[valid_operator_combination_list.index(chosen_combination)] * factor)
          ag.task_method_hierarchy.append(update_oper)
          ag.prob_hierarchy.append(1)

      elif update_oper not in ag.task_method_hierarchy and ag.name not in agent_reach_action:
        if len(ag.task_method_hierarchy) > 0:
          label_last_ele = check_task_method_action(world.env_dictionary, ag.task_method_hierarchy[-1])
          label_oper = check_task_method_action(world.env_dictionary, update_oper)
          if label_last_ele == 'task' and label_oper == 'task':
            # remove the last ele
            ag.task_method_hierarchy.pop(-1)
            ag.prob_hierarchy.pop(-1)
          elif label_last_ele == 'method' and label_oper == 'method':
            # remove both the method and the coresponding task:
            ag.task_method_hierarchy.pop(-1)
            ag.prob_hierarchy.pop(-1)
            corresponding_task_oper = find_corresponding_task(world.env_dictionary, update_oper)
            # continue removing the task from the hierarchy if not matching with the corresponding task of the udpate_oper:
            if len(ag.task_method_hierarchy) > 0:
              if corresponding_task_oper != ag.task_method_hierarchy[-1]:
                ag.task_method_hierarchy.pop(-1)
                ag.prob_hierarchy.pop(-1)
                ag.task_method_hierarchy.append(corresponding_task_oper)
                ag.prob_hierarchy.append(1)
              # else, no more removing
          elif label_last_ele == 'task' and label_oper == 'method':
            # if the last ele task is not the same as corresponding task of the method, replace with the corresponding task
            corresponding_task_oper = find_corresponding_task(world.env_dictionary, update_oper)
            if corresponding_task_oper != ag.task_method_hierarchy[-1]:
              ag.task_method_hierarchy[-1] = corresponding_task_oper
              ag.prob_hierarchy[-1] = 1
            # else, do nothing
          elif label_last_ele == 'task' and label_oper == 'action':
            # simply remove the task before append the action:
            ag.task_method_hierarchy.pop(-1)
            ag.prob_hierarchy.pop(-1)
            agent_reach_action.append(ag.name)
          elif label_last_ele == 'action' and label_oper == 'action': #in case for some reason the agent_reach_action is not updated yet
            # print("CCP 627: agent_reach_action is not updated properly, the agent {} (is belief: {}) should be in the list".format(ag.name, ag.belief_flag))
            ag.task_method_hierarchy.pop(-1)
            ag.prob_hierarchy.pop(-1)
            agent_reach_action.append(ag.name)
        
        # print("*Update hierarchy with new oper:", update_oper)
        ag.task_method_hierarchy.append(update_oper)
        ag.prob_hierarchy.append(valid_prob_list[valid_operator_combination_list.index(chosen_combination)] * factor)
      elif update_oper in ag.task_method_hierarchy and update_oper != ag.task_method_hierarchy[-1]:
        if chosen_combination.index(update_oper) == index_agent:
          ag.task_method_hierarchy.append(update_oper)
          ag.prob_hierarchy.append(valid_prob_list[valid_operator_combination_list.index(chosen_combination)] * factor)

      # 5.4. update the agent_reach_action list according to the current hierarchies
      if len(ag.task_method_hierarchy)>0:
        last_ele_label = check_task_method_action(world.env_dictionary, ag.task_method_hierarchy[-1])
        if last_ele_label == 'action' and ag.name not in agent_reach_action:
          agent_reach_action.append(ag.name)
        elif last_ele_label != 'action' and ag.name in agent_reach_action:
          agent_reach_action.remove(ag.name)
    
  return agents
