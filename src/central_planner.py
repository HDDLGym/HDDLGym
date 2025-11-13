# MODIFY if needed: Change "learning_methods" to any learning methods file that you design:
from learning_methods import get_grounded_prob_list_from_policy_output, get_observation_one_hot_vector, get_probabilities_of_operators
from learning_methods import get_probability_of_valid_operator_combinations, pick_policy

from central_planner_utils import generate_valid_operators, check_valid_combination_operator
from hddl_utils import find_corresponding_task, check_task_method_action

import numpy as np
import random
import itertools
import copy
import torch
import time


def centralized_planner(world_info, main_agent_index=0, all_agents = [], all_policies = [], debug=False, deterministic=False, device = False, time_limit = 60):
  ''' This function plan the next operators of all agents up until all of them reach actions
  inputs:
  - world_info: using info of world_info.num_agents, world_info.current_state, world_info.env_ditionary dict
  - all_agents: list of instances of Agent, if empty (default), set it to copy.deepcopy(world_info.agents)
  - all_policies: list of policy dictionaries of all agents, if empty (default), use random policy for all of them
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
  overall_start_time = time.time()
  time_get_prob = 0
  time_get_prob_old = 0
  overall_run_time_old = 0
  time_choosing = 0
  time_generate_valid_oper = 0
  time_update_hierarchy = 0
  # 0. Common parameters:
  if not device:
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  infeasible_com_list = list()
  assert len(all_agents) == world_info.num_agents, \
    "Centralized_planner: Number of agents in the world_info ({}) and \
      the number of agents in the all_agents list ({}) are not equal!".format(world_info.num_agents,\
                                                                               len(all_agents))
  # agents = all_agents

  if len(all_policies) == 0:
    all_policies = [None] * world_info.num_agents
  # main_policy_dict = all_policies[main_agent_index]
  # state = world_info.current_state
  
  # List of all operators for observation
  all_operators = []
  all_current_valid_operators = set()
  for agent in all_agents:
    all_operators += agent.prev_task_method_hierarchy
    if agent.htn_goal != None:
      all_operators += agent.htn_goal.pending_tasks
      all_operators += agent.htn_goal.remaining_tasks
  all_operators = set(all_operators)
  all_operators = list(all_operators)
  observation_num = get_observation_one_hot_vector(world_info.current_state, all_operators, world_info.env_dictionary, device = device)
  agent_reach_action = [] # list of name of agents who have reach their action in their task_method_hierarchy
  if debug: print("---Start while loop...")
  time_start = time.time()
  while len(agent_reach_action) < world_info.num_agents:
    elapsed_time = time.time() - time_start
    if elapsed_time > time_limit:
      if debug:
        print("Centralized Planner's running time has reach the time limit of {} seconds, setting incomplete agents' hierarchy to ['none']!".format(time_limit))
      for agent_index, agent in enumerate(all_agents):
        if agent.name not in agent_reach_action:
          agent.task_method_hierarchy = ['none '+agent.name]
          agent.prob_hierarchy = [1]
      break # break the while loop
    valid_operators_dict = dict()
    prob_valid_operators_dict = dict()
    # 1. Find a valid list of operator of the agent, giving the current hierarchy, use policy to find prob list of them
    current_hierachy = []
    for i_a1, agent in enumerate(all_agents):
      if len(agent.task_method_hierarchy) > 0:
        current_hierachy.append(agent.task_method_hierarchy[-1])
      if agent.name in agent_reach_action:
        valid_operators_dict[agent.name] = [agent.task_method_hierarchy[-1]]
        # print("CCP377: valid oper is copied for {} with last element of hierarchy {}".format(agent.name, agent.task_method_hierarchy[-1]))
      else: # agent.name not in agent_reach_action
        has_valid = False
        while not has_valid: #keep modifying the hierarchy until has valid following oper
          # print("DEBUG generate_valid_op: current agent infeasible list:", agent.infeasible_task_method)
          agent.current_valid_operators.update(all_current_valid_operators)
          start_time = time.time()
          valid_op_list, empty_by_others, current_valid_operators = generate_valid_operators(world_info, agent, print_empty_by_others=True)
          # print("---Time of run generate_valid_operators: ", time.time()-start_time)
          time_generate_valid_oper += time.time() - start_time
          all_current_valid_operators.update(agent.current_valid_operators)

          # HANDLE THE CASE WHEN THE AGENT HAS NO VALID OPERATOR:
          # Only accept None action if no valid operator is due to other collaborative agents
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
            # print("DEBUG generate_valid_op: after trying generate valid oper, agent infeasible list:", agent.infeasible_task_method)
            # Remove this infeasible task method from the all_current_valid_operators:
            all_current_valid_operators.discard(agent.task_method_hierarchy[-1])
            # If the infeasible task method is in other all_agents' hierarchy, remove them too:
            for other_agent_ in all_agents:
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
            # If the agent is doing the task in the htn goal, while performing other goals, make sure to mark it in main agent's htn goal
            if agent.task_method_hierarchy[-1] in all_agents[main_agent_index].htn_goal.pending_tasks:
              all_agents[main_agent_index].htn_goal.remaining_tasks.append(agent.task_method_hierarchy[-1])
              all_agents[main_agent_index].htn_goal.pending_tasks.remove(agent.task_method_hierarchy[-1])
            # Now, since the agent has no valid operator, remove the last element of the hierarchy before re-generate the valid operator list 
            agent.task_method_hierarchy.pop(-1)
            agent.prob_hierarchy.pop(-1)

          # HANDLE THE CASE WHEN THE AGENT HAS VALID OPERATOR:
          elif len(valid_op_list) != 0:
            # if there are actions in the valid op list, add 'none' to the list
            for op in valid_op_list:
              if check_task_method_action(world_info.env_dictionary, op) == 'action':
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

    # Now, use the policy to get the probability of the valid operators for the agent:
    start_time_get_prob = time.time()
    for id_prob, agent_get_prob in enumerate(all_agents):
      policy = pick_policy(all_policies[id_prob], world_info.env_dictionary, current_hierarchy=agent_get_prob.task_method_hierarchy, preferences="all operators")
      prob_valid_operators_dict[agent_get_prob.name] = get_probabilities_of_operators(valid_operators_dict[agent_get_prob.name], agent_get_prob, world_info.current_state, world_info.env_dictionary, policy=policy) 
    time_get_prob += time.time() - start_time_get_prob

    # 2. Create combination list of all possible comb from the lists of opers, and validate each combination
    # Generate combinations of operators 
    operator_options_list = list(valid_operators_dict[agent.name] for agent in all_agents)
    operator_combination_list = list(itertools.product(*operator_options_list))
    # Generate combinations of probabilities
    start_time_get_prob = time.time()
    # print("prob_valid_operators_dict", prob_valid_operators_dict)
    # print("valid_operators_dict", valid_operators_dict)
    prob_operator_options_list = list(prob_valid_operators_dict[agent.name].cpu().tolist() for agent in all_agents)
    prob_operator_combination_list = []
    # print("comb of prob:", list(itertools.product(*prob_operator_options_list)))
    # print("comb of oper:", operator_combination_list)
    for prob_comb in itertools.product(*prob_operator_options_list):
        # Calculate the product of probabilities for the combination
        prob_comb_tensor = torch.tensor(prob_comb)  # Create a tensor from the tuple
        prob_operator_combination_list.append(prob_comb_tensor.prod().item())  # Convert to a float and append to the list
    time_get_prob += time.time() - start_time_get_prob

    # Change the strategy: choose a combination from the probability list, then check if the combination is valid, if not, remove it from the list, and repeat until get one.
    start_time_choosing = time.time()
    choosing_combination_list = copy.deepcopy(operator_combination_list)
    choosing_prob_list = copy.deepcopy(prob_operator_combination_list)
    choosing_prob_list = torch.tensor(choosing_prob_list, device=device)
    if len(choosing_combination_list) != len(choosing_prob_list):
        print("prob_valid_operators_dict", prob_valid_operators_dict)
        print("valid_operators_dict", valid_operators_dict)
        raise ValueError(f"Before while loop: The number of weights, {len(choosing_prob_list)}, does not match the population, {len(choosing_combination_list)}.\n choosing_com_list {choosing_combination_list}, \nchoosing_prob_list {choosing_prob_list}")

    found_valid_combination = False
    continue_big_while = False

    while len(choosing_combination_list) > 0 and not found_valid_combination:
      if deterministic:
        chosen_combination = choosing_combination_list[torch.argmax(choosing_prob_list).item()]
        prob_chosen_combination = choosing_prob_list[torch.argmax(choosing_prob_list).item()]
      else:
        # Ensure weights and population lengths match
        if len(choosing_combination_list) != len(choosing_prob_list):
            raise ValueError(f"The number of weights, {len(choosing_prob_list)}, does not match the population, {len(choosing_combination_list)}.")

        # print("choose randomly with weights are prob")
        chosen_combination = random.choices(choosing_combination_list, weights = choosing_prob_list.tolist(), k=1)[0]
        prob_chosen_combination = choosing_prob_list[choosing_combination_list.index(chosen_combination)]

      if check_valid_combination_operator(world_info.current_state, world_info.env_dictionary, chosen_combination, all_agents, infeasible_com_list = infeasible_com_list):
        found_valid_combination = True
      else:
        index_to_remove = choosing_combination_list.index(chosen_combination)
        choosing_combination_list.pop(index_to_remove)
        # Remove the probability by slicing the tensor
        choosing_prob_list = torch.cat((choosing_prob_list[:index_to_remove], choosing_prob_list[index_to_remove + 1:]))
        
        if len(choosing_combination_list) == 0:
          # Handle the case when there is nothing in valid_operator_combination_list:
          if debug:
            print('No valid combination amongst the list:',operator_combination_list)
          # remove the last layer of the hierarchies of all_agents and append them to the infeasible and remove any agents from reach_action_agents
          current_com = []
          for a__ in all_agents:
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
          continue_big_while = True

      if debug and len(choosing_combination_list) >=2 and 'none' in ' '.join(chosen_combination):
        print("\n***chosen com (centralized_planner) {} from list \n {} \n with prob {}".format(chosen_combination, choosing_combination_list,choosing_prob_list))
    time_choosing += time.time() - start_time_choosing
    if continue_big_while:
      continue

    '''
    # 2.1. prune invalid combination:
    start_pruning_time = time.time()
    valid_operator_combination_list = []
    valid_prob_list = []
    start_comb_time = time.time()
    for index_com, com in enumerate(operator_combination_list):
      start_time = time.time()
      if check_valid_combination_operator(world_info.current_state, world_info.env_dictionary, com, all_agents, infeasible_com_list = infeasible_com_list):
        valid_operator_combination_list.append(com)
        valid_prob_list.append(prob_operator_combination_list[index_com])
    #   print("--- Time checking valid operator:", time.time()-start_time)
    # print("--- Time of checking all valid_combination_operator: ", time.time()-start_comb_time)
    # Handle the case when there is nothing in valid_operator_combination_list:
    overall_run_time_old += time.time() - start_pruning_time
    if len(valid_operator_combination_list) == 0:
      if debug:
        print('No valid combination amongst the list:',operator_combination_list)
      # remove the last layer of the hierarchies of all_agents and append them to the infeasible and remove any agents from reach_action_agents
      current_com = []
      for a__ in all_agents:
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

    start_get_prob_time = time.time()
    valid_prob_list = get_probability_of_valid_operator_combinations(valid_operator_combination_list, \
                                                                    all_policies, \
                                                                    main_agent_index, \
                                                                    world_info.current_state, \
                                                                    world_info.env_dictionary, \
                                                                    all_agents, \
                                                                    device=device)
    
    time_get_prob_old += time.time()-start_get_prob_time
    overall_run_time_old += time.time()-start_get_prob_time
    # print("\n--- Time of getting prob list of valid operator combinations: ", time_get_prob_old)

    # 4. choose a combination by random choice with prob:
    start_choosing_time_old = time.time()
    if deterministic:
      _chosen_combination = valid_operator_combination_list[torch.argmax(valid_prob_list).item()]
      _prob_chosen_combination = valid_prob_list[torch.argmax(valid_prob_list).item()]
    else:
      # print("choose randomly with weights are prob")
      _chosen_combination = random.choices(valid_operator_combination_list, weights = valid_prob_list, k=1)[0]
      _prob_chosen_combination = valid_prob_list[valid_operator_combination_list.index(chosen_combination)]
    
    if debug and len(valid_operator_combination_list) >=2 and 'none' in ' '.join(chosen_combination):
      print("\n***chosen com (CCP 465, centralized_planner) {} from list \n {} \n with prob {}".format(chosen_combination, valid_operator_combination_list,valid_prob_list))
    overall_run_time_old += time.time() - start_choosing_time_old
    '''
    
    
    # 5. Update the hierarchy of the agents: for each agent:
    start_time_update_hierarchy = time.time()
    for index_agent, ag in enumerate(all_agents):
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
      elif (len(related_oper_indices) == 2 and 'none '+ag.name in chosen_combination and world_info.num_agents<=2):
        #Prefer to choose operator other than none action. Note that this won't apply for 3+ agents
        not_none_index = int(1-chosen_combination.index('none '+ag.name))
        update_oper = chosen_combination[not_none_index]

      # 5.2. if agent appears in more than 2 opers: use agent's policy to choose (random.choices with weight is prob from the policy)
      elif len(related_oper_indices) >= 2:
        agent_policy_dict = all_policies[index_agent]
        agent_policy = pick_policy(agent_policy_dict, world_info.env_dictionary, current_hierarchy=all_agents[index_agent].task_method_hierarchy, preferences="all operators")
        if agent_policy == None: #uniform probability when policy is None
          prob_op_list = torch.tensor([1/len(related_oper_indices)]*len(related_oper_indices)) #make sure the prob list is a tensor
        else:
          agent_prob_oper_list = agent_policy.select_action(observation_num, value=True)
          prob_op_list = []
          related_opers = [chosen_combination[com_i] for com_i in related_oper_indices]
          prob_op_list = get_grounded_prob_list_from_policy_output(related_opers, agent_prob_oper_list, world_info.env_dictionary, device=device)
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
        label_oper = check_task_method_action(world_info.env_dictionary, update_oper)
        if label_oper == 'action' and len(ag.task_method_hierarchy)>0:
          # print("pop out the current action {} before adding new action {}".format(ag.task_method_hierarchy),update_oper)
          ag.task_method_hierarchy.pop(-1)
          ag.prob_hierarchy.pop(-1)
          ag.task_method_hierarchy.append(update_oper)
          ag.prob_hierarchy.append(prob_chosen_combination * factor)
          # print("new hierarchy after replace action: ", ag.task_method_hierarchy)
        elif label_oper == 'task':
          agent_reach_action.remove(ag.name)
          ag.task_method_hierarchy.pop(-1)
          ag.prob_hierarchy.pop(-1)
          # print("Update hierarchy with new oper:", update_oper)
          ag.task_method_hierarchy.append(update_oper)
          ag.prob_hierarchy.append(prob_chosen_combination * factor)
        elif label_oper == 'method':
          agent_reach_action.remove(ag.name)
          ag.task_method_hierarchy.pop(-1)
          ag.prob_hierarchy.pop(-1)
          corresponding_task = find_corresponding_task(world_info.env_dictionary, update_oper)
          ag.task_method_hierarchy.append(corresponding_task)
          ag.prob_hierarchy.append(prob_chosen_combination * factor)
          ag.task_method_hierarchy.append(update_oper)
          ag.prob_hierarchy.append(1)

      elif update_oper not in ag.task_method_hierarchy and ag.name not in agent_reach_action:
        if len(ag.task_method_hierarchy) > 0:
          label_last_ele = check_task_method_action(world_info.env_dictionary, ag.task_method_hierarchy[-1])
          label_oper = check_task_method_action(world_info.env_dictionary, update_oper)
          if label_last_ele == 'task' and label_oper == 'task':
            # remove the last ele
            ag.task_method_hierarchy.pop(-1)
            ag.prob_hierarchy.pop(-1)
          elif label_last_ele == 'method' and label_oper == 'method':
            # remove both the method and the coresponding task:
            ag.task_method_hierarchy.pop(-1)
            ag.prob_hierarchy.pop(-1)
            corresponding_task_oper = find_corresponding_task(world_info.env_dictionary, update_oper)
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
            corresponding_task_oper = find_corresponding_task(world_info.env_dictionary, update_oper)
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
        ag.prob_hierarchy.append(prob_chosen_combination * factor)
      elif update_oper in ag.task_method_hierarchy and update_oper != ag.task_method_hierarchy[-1]:
        if chosen_combination.index(update_oper) == index_agent:
          ag.task_method_hierarchy.append(update_oper)
          ag.prob_hierarchy.append(prob_chosen_combination * factor)

      # 5.4. update the agent_reach_action list according to the current hierarchies
      if len(ag.task_method_hierarchy)>0:
        last_ele_label = check_task_method_action(world_info.env_dictionary, ag.task_method_hierarchy[-1])
        if last_ele_label == 'action' and ag.name not in agent_reach_action:
          agent_reach_action.append(ag.name)
        elif last_ele_label != 'action' and ag.name in agent_reach_action:
          agent_reach_action.remove(ag.name)
    time_update_hierarchy += time.time() - start_time_update_hierarchy
    
  overall_run_time = time.time() - overall_start_time
  if debug:
    print("\nOverall run time of the centralized planner: ", overall_run_time)
    print(f"Time getting prob: {time_get_prob}, {time_get_prob/overall_run_time*100:.3f}% total")
    print(f"Time choosing: {time_choosing}, {time_choosing/overall_run_time*100:.3f}% total")
    print(f"Time generating valid operators: {time_generate_valid_oper}, {time_generate_valid_oper/overall_run_time*100:.3f}% total")
    print(f"Time updating hierarchy: {time_update_hierarchy}, {time_update_hierarchy/overall_run_time*100:.3f}% total")
  return all_agents
