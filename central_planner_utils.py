import numpy as np
from hddl_utils import extract_object_list, Task, Method, check_subtask_with_world_state
from learning_methods import get_probabilities_of_operators
import copy
import random
import numpy as np

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


def choose_method_policy(env_dictionary, state, task, agent, method_policy = None, print_valid_list = False, debug=False, collaborative=False, collab_agents = []):
  '''policy of choosing method for the task
  input:
  - env_dictionary: environment dictionary, has info about task, method, action, etc.
  - state: list of predicates indicating the current state of the world
  - task: string of task and objects, e.g. 'deliver package-1 city-2'
  - agent_name: agent's name, indicating this agent is performing the task
  outputs:
  - method
  - if print_valid_list: list of valid grounded methods
  '''
  # debug=True
  #1. list all lifted methods can be used for the task:
  task_name, task_objects_list = extract_object_list(task)
  agent_in_object = agent.name in task_objects_list
  matched_task = None
  for lifted_task in env_dictionary['lifted tasks list']:
    if lifted_task.name == task_name:
      matched_task = lifted_task
      break
  assert isinstance(matched_task, Task), "couldn't find task {} in the env task list".format(task_name)

  lifted_methods_list = matched_task.related_methods_list
  # 2. list all grounded methods can be used for the task
  grounded_methods_list = []
  # modify type_object_dict so only the agent involve in the non-collaborative task:
  type_object_dict = copy.deepcopy(env_dictionary['type object dict'])
  
  for lifted_method in lifted_methods_list:
    if debug:
      print('lifted method for this task {}: {}'.format(task_name, lifted_method.name))
    grounded_methods_list.append(lifted_method.generate_grounded_methods(type_object_dict,\
                                                             task_var_object_dict = lifted_method.translate_objects_list_to_dict(task_objects_list)))
  # if debug:
  #   print("CP 197: before pruning not related grounded method, grounded_methods_list:", grounded_methods_list)
  #Prune grounded methods that are not related to agent:
  other_agents_list = copy.deepcopy(env_dictionary['type object dict']['agent'])
  other_agents_list.remove(agent.name)
  invalid_by_agent_grounded_methods = []
  for grounded_method_group in grounded_methods_list:
    for grounded_method in grounded_method_group:
      if agent.name not in grounded_method:
        for other_agent in other_agents_list:
          if other_agent in grounded_method:
            invalid_by_agent_grounded_methods.append(grounded_method)
  for invalid_grounded_method in invalid_by_agent_grounded_methods:
    for grounded_method_group_index, grounded_method_group in enumerate(grounded_methods_list):
      if invalid_grounded_method in grounded_method_group:
        grounded_methods_list[grounded_method_group_index].remove(invalid_grounded_method)

  if debug:
    print("central_planner_utils.py: choose method policy: AFTER pruning not related grounded method, \
      grounded_methods_list for task {}:\n {}".format(task,grounded_methods_list))
  ###
  #3. prune invalid methods: by checking precondition
  valid_grounded_methods = []
  for i in range(len(grounded_methods_list)):
    for j in range(len(grounded_methods_list[i])):
      name, obj_list = extract_object_list(grounded_methods_list[i][j])
      assert name == lifted_methods_list[i].name, "name of grounded and lifted don't match!"
      var_obj_dict = {lifted_methods_list[i].parameters_name_list[k] : obj_list[k] for k in range(len(obj_list))}
      if lifted_methods_list[i].precondition.check_precondition(state, var_obj_dict, debug=debug) and grounded_methods_list[i][j] not in agent.infeasible_task_method:
        valid_grounded_methods.append(grounded_methods_list[i][j])

  if debug:
    print('Central_planner_utils.py: choose method policy: valid grounded methods for task {} is {}'.format(task, valid_grounded_methods))
  #3b. if print_valid_list is True: return valid_grounded_methods:
  if print_valid_list:
    # print('valid grounded methods for task {} is {}'.format(task, valid_grounded_methods))
    return valid_grounded_methods

  #4 choose a method from the valid grounded methods list:
  if len(valid_grounded_methods) == 0:
    agent.infeasible_task_method.append(task)
    return None, 0
  if method_policy == None:
    #pick randomly:
    chosen_method = random.choice(valid_grounded_methods)
    prob_method = 1/len(valid_grounded_methods)
  elif len(valid_grounded_methods)==1:
    chosen_method = valid_grounded_methods[0]
    prob_method = 1
  else:
    probability_list_of_valid_methods = get_probabilities_of_operators(valid_grounded_methods, agent, state, env_dictionary,method_policy)
    chosen_method = random.choices(valid_grounded_methods, weights = probability_list_of_valid_methods.tolist(), k=1)[0]
    prob_method = policy_values[valid_grounded_methods.index(chosen_method)]

  #5. return chosen method:
  if debug:
    print("chosen method for task {} is {} with prob {}".format(task, chosen_method,prob_method))
  return chosen_method, prob_method



def choose_subtask_policy(env_dictionary, state, method_str, agent, completed_subtasks= [], subtask_policy=None, print_valid_list = False, collaborative=False, collab_agents=[], print_empty_by_others=False, debug=False):
  ''' choose subtask to perform the method in given state
  inputs:
  - env_dictionary: environment of dictionary
  - state
  - method_str: a string of <method_name> <object parameters>, e.g. 'm-drive truck-0 city-0 city-1'
  - agent
  - completed_subtasks
  - subtasks_policy
  - print_valid_list: boolean flag to indicate we only need list of valid subtasks, no need to choose a specific one with policy
  - collaborative: boolean flag indicating whether the method is collaborative or not
  - collab_agent: the agent that this agent is collaborating with in this collaborative method
  - print_empty_by_others : a boolean to indicate if the function should return the flag of whether no valid subtask is due to other agents' progresses
  output:
  - subtask (can be either task or action)

  Approach:
  1. using info from instance method to have a list of subtasks
  2. prune invalid valid subtask (task or action) by comparing orders and precondition with state
  3. choose subtask from the valid subtask list:
  4. return subtask or a list of valid subtasks
  '''
  # debug=True
  # 1. using info from instance method to have a list of subtasks
  method_name, method_objects = extract_object_list(method_str)
  agent_in_object = agent.name in method_objects
  method = None
  for lifted_method in env_dictionary['lifted methods list']:
    if lifted_method.name == method_name:
      method = lifted_method
      break
  assert isinstance(method, Method), "cannot find method {} from the method list".format(method_str)
  subtasks_list = copy.copy(method.subtasks)
  ordering_list = copy.copy(method.ordering)

  #Note: subtasks_list is a list of string of <subtask_name> <var_name>...
  # Ground the subtasks_list (replace var_name with object)

  for i, param_name in enumerate(method.parameters_name_list):
    for j, sub in enumerate(subtasks_list):
      subtasks_list[j] = sub.replace(param_name, method_objects[i])
    for k, (t1,t2) in enumerate(ordering_list):
      ordering_list[k] = (t1.replace(param_name, method_objects[i]),t2.replace(param_name, method_objects[i]))


  # Prune subtasks that are not related to the agent:
  other_agents_list = copy.deepcopy(env_dictionary['type object dict']['agent'])
  other_agents_list.remove(agent.name)
  
  # Remove subtask that are completed given the world state
  completed_grounded_subtasks = check_subtask_with_world_state(env_dictionary, state, subtasks_list)
  if debug:
    print('choose subtask policy: completed grounded subtask for agent {} with method_str {}: {}'.format(agent.name,method_str, completed_grounded_subtasks))
  for completed_subtask in completed_grounded_subtasks:
    if completed_subtask in subtasks_list:
      subtasks_list.remove(completed_subtask)
  
  # Removed subtasks that are meant for other agents (in collaborative methods)
  invalid_by_agent_subtask = []
  for subtask in subtasks_list:
    if agent.name not in subtask:
      for other_agent in other_agents_list:
        if other_agent in subtask:
          invalid_by_agent_subtask.append(subtask)
  for invalid_subtask in invalid_by_agent_subtask:
    if invalid_subtask in subtasks_list:
      subtasks_list.remove(invalid_subtask)

  # 2. prune invalid valid subtask (task or action) by comparing order and precondition with state
  # if subtask is primitive action: check precondition, if subtask is task, check precondition of all possible methods
  subtasks_instance_list = []
  valid_subtasks_str = []
  valid_subtasks_instance = []
  if debug:
    print("-------- valid subtask list:", subtasks_list)

  for sub in subtasks_list:
    sub_name, sub_objects_list = extract_object_list(sub)
    # make sure the subtask sub is related to agent, otw, subtasks shouldn't include any agents:
    not_related =False
    if agent.name not in sub_objects_list:
      for a in collab_agents:
        if a.name in sub_objects_list:
          not_related = True
          break
    if not_related:
      continue
    sub_is_action = False
    sub_is_task = False
    # if sub is action:
    for action in env_dictionary['lifted actions list']:
      if action.name == sub_name:
        sub_is_action = True
        subtasks_instance_list.append(action)
        # check validity:
        var_object_dict = action.translate_objects_list_to_dict(sub_objects_list)
        if action.precondition.check_precondition(state,var_object_dict,debug=debug) and sub not in agent.completed_tasks_agent:
          valid_subtasks_str.append(sub)
          valid_subtasks_instance.append(action)

        break
    if not sub_is_action:
      for task in env_dictionary['lifted tasks list']:
        if task.name == sub_name:
          sub_is_task = True
          subtasks_instance_list.append(task)
          _m_list = choose_method_policy(env_dictionary, state, sub, agent, method_policy=None, print_valid_list=True, debug=False)
          if debug:
            print('_m_list {} for subtask {} while choosing subtask for method {}:'.format(_m_list, sub_name, method_str))
          if len(_m_list) >0:# and _m not in agent.infeasible_task_method and sub not in agent.completed_tasks_agent:
            valid_subtasks_instance.append(task)
            valid_subtasks_str.append(sub)
          # else:
          #   print("CP 374: agent {} task {} is not valid bc no method can do it!".format(agent.name,sub))
          break
  if debug:
    print("** central_planner: choose subtask policy: Before check ordering of choose_subtask_method, valid subtasks str:",valid_subtasks_str)

  #also need to check ordering
  invalid_by_order = []
  invalid_by_other_agent = []
  completed_subtasks = set(completed_subtasks)# + completed_grounded_subtasks)
  completed_subtasks.update(agent.completed_tasks_agent)
  if len(collab_agents) > 0:
    for c_a in collab_agents:
      completed_subtasks.update(c_a.completed_tasks_agent)
  # print("-----completed subtasks: ", completed_subtasks)
  after_ordering_subtasks_str = copy.copy(valid_subtasks_str)
  invalid_by_order = get_invalid_by_order_subtasks(completed_subtasks, ordering_list,valid_subtasks_str)
  for t in valid_subtasks_str:
    if t in invalid_by_order or t in completed_subtasks:
      after_ordering_subtasks_str.remove(t)

  after_ordering_other_agent = copy.deepcopy(after_ordering_subtasks_str)
  len_before_other_agent = len(after_ordering_other_agent)
  for t_after in after_ordering_subtasks_str:
    if t_after in invalid_by_other_agent:
      after_ordering_other_agent.remove(t_after)

  empty_by_others = False
  if len(after_ordering_other_agent) == 0 and len_before_other_agent >0:
    empty_by_others = True

  if debug:
    print('Central_planner: choose subtask policy: after ordering subtasks str:', after_ordering_subtasks_str)
  if len(after_ordering_other_agent) == 0 and len(after_ordering_subtasks_str) > 0:
    after_ordering_other_agent.append('none {}'.format(agent.name))

  if print_valid_list:
    if print_empty_by_others:
      # print('after ordering other agent:',after_ordering_other_agent)
      return after_ordering_other_agent, empty_by_others
    return after_ordering_other_agent

  # 3. choose subtask from the valid subtask list
  if len(after_ordering_subtasks_str) == 0:
    agent.infeasible_task_method.append(method_str)
    if print_empty_by_others:
      return None, 0, empty_by_others
    return None, 0
  elif subtask_policy == None:
    if print_empty_by_others:
      return random.choice(after_ordering_other_agent), 1/len(after_ordering_other_agent), empty_by_others
    return random.choice(after_ordering_other_agent), 1/len(after_ordering_other_agent)
  else:
    probability_list_of_valid_subtasks = get_probabilities_of_operators(after_ordering_other_agent, agent, state, env_dictionary, subtask_policy)
    chosen_subtask = random.choices(after_ordering_other_agent, weights = probability_list_of_valid_subtasks.tolist(), k=1)[0]
    prob_s = policy_values[after_ordering_other_agent.index(chosen_subtask)]
    if print_empty_by_others:
      return chosen_subtask, prob_s, empty_by_others
    return chosen_subtask, prob_s


def generate_valid_operators(world, agent, print_empty_by_others = False):
  '''This function is to find a list of all valid possible operators of the agent,
  given the current state of the world and the task_method_hierarchy
  inputs:
  - world: an HDDLEnv instance
  - agent: an Agent instance, whose task_method_hierarchy[-1] should be either task or method or empty
  output:
  - valid_op_list: list of valid grounded operators
  - empty_by_others
  '''
  valid_op_list = []
  empty_by_others = False
  if len(agent.task_method_hierarchy) == 0:
    # for t in world.env_dictionary['htn goal'].remaining_tasks:
    # print("CCP 552 - generate valid oper: start of the hierarchy, the remaining goal task to assign to agent {}: {}".format(agent.name, agent.htn_goal.remaining_tasks))
    for t in agent.htn_goal.remaining_tasks: #6.25
      if t not in agent.infeasible_task_method:
        valid_op_list.append(t)
    #Also append 'none' into valid_op_list: for the case agent doesn't want to do anything
    valid_op_list.append('none '+agent.name)
    # if len(valid_op_list) == 0: #No more goal tasks or feasible goal tasks to assign:
    #   valid_op_list.append('none {}'.format(agent.name))
  elif check_task_method_action(world.env_dictionary, agent.task_method_hierarchy[-1]) == 'task':
    collaborative, collab_agent_name = check_collaborative_status(world.env_dictionary, agent, agent.task_method_hierarchy[-1])
    collab_agents = []
    if len(collab_agent_name) > 0:
      for other_agent in agent.belief_other_agents:
        if other_agent.name in collab_agent_name:
          collab_agents.append(other_agent)
    valid_op_list = choose_method_policy(world.env_dictionary, world.current_state, agent.task_method_hierarchy[-1], agent, method_policy=agent.agent_policy, \
                                         print_valid_list = True, collaborative=collaborative, collab_agents = collab_agents, debug=False)
    if len(valid_op_list) == 0:
      # Cannot find a valid method to perform the task ==> the task is infeasible:
      agent.infeasible_task_method.append(agent.task_method_hierarchy[-1])
      
    
  elif check_task_method_action(world.env_dictionary, agent.task_method_hierarchy[-1]) == 'method':
    collaborative, collab_agent_name = check_collaborative_status(world.env_dictionary, agent, agent.task_method_hierarchy[-1])
    collab_agents = []
    if len(collab_agent_name) > 0:
      for other_agent in agent.belief_other_agents:
        if other_agent.name in collab_agent_name:
          collab_agents.append(other_agent)
    
    valid_op_list, empty_by_others = choose_subtask_policy(world.env_dictionary, world.current_state, method_str=agent.task_method_hierarchy[-1], agent=agent, subtask_policy = agent.agent_policy,\
                                          print_valid_list=True, collaborative=collaborative, collab_agents = collab_agents, print_empty_by_others = print_empty_by_others, debug=False)
    
    # if 'none {}'.format(agent.name) not in valid_op_list:
    if len(valid_op_list) == 0:
      valid_op_list.append('none {}'.format(agent.name))
  else:
    # print("CCP 570: The last operator is neither task nor method! Please check: last operator: \n{}".format(agent.task_method_hierarchy[-1]))
    valid_op_list = [agent.task_method_hierarchy[-1]]
  if print_empty_by_others:
    return valid_op_list, empty_by_others
  return valid_op_list


def check_valid_combination_operator(current_state, env_dictionary, combination, agents_list, infeasible_com_list = [], debug=False):
  '''This help check if the everything is perform parallelly, will it cause conflict
  This only help checking:
  1. each agent only involves in exactly 1 operator, Note that the operator not neccessarily include agent.name in it (goal tasks)
  2. no effect in action conflict to the precondition of other action/method. (effect of task is ok)

  inputs:
  current_state: list of grounded predicates
  env_dictionary: environment dictionary
  combination: tuple or list of strings of operators
  agents_list: list of agents
  infeasible_com_list: list of set of infeasible combination found (so no need to dive deeper)
  - output:
  valid: a boolean indicate if the combination is value or not
  '''
  valid = True
  if set(combination) in infeasible_com_list:
    return False
  world_state_list = copy.deepcopy(current_state) # will modify the world state but don't want to affect the real world
  for index_agent, agent in enumerate(agents_list):
    count = 0
    agent_oper = set()
    for index_operator, oper in enumerate(combination):
      if index_agent == index_operator and oper != 'none {}'.format(agent.name):
        agent_oper.add(oper)
      elif index_agent != index_operator and agent.name in oper:
        # only concern if the oper is actual action:
        if check_task_method_action(env_dictionary, oper) == 'action':
          agent_oper.add(oper)
    if len(agent_oper) > 1:
      # print("invalid com {} because agent {} appear {} times".format(combination, agent.name, count))
      valid = False
      return valid

  # 2. check if the effect of any action (if exist) is conflicted to precondition of other methods/actions
  need_check_precondition_list = []
  need_check_effect_list = []
  for oper in combination:
    label = check_task_method_action(env_dictionary, oper)
    if label == 'action':
      temp_state = copy.deepcopy(current_state)
      new_temp_state = apply_action_effect(env_dictionary, oper, temp_state)
      for other_oper in combination:
        if other_oper != oper:
          if check_task_method_action(env_dictionary, other_oper) in ['action','method']:
            if not check_precondition(env_dictionary, other_oper, new_temp_state):
              valid = False
              if debug:
                print("{} become invalid after {} is applied".format(other_oper, oper))
              return valid

  # make sure all effect is in the
  if debug:
    print("valid: ",valid)
  return valid

def apply_action_effect(env_dictionary, action_str, state):
  '''This function apply effect of the action to the state
  inputs:
  - env_dictionary: a dictionary of HDDL parser
  - action_str: action string, include name and objects (grounded)
  - state: list of predicates
  outputs:
  - new_state: new list of predicates

  '''
  new_state = state
  # 1. look up lifted action instance of action_str:
  action_name, action_object = extract_object_list(action_str)
  matched_action = None
  for lifted_action in env_dictionary['lifted actions list']:
    if lifted_action.name == action_name:
      matched_action = lifted_action
  assert matched_action != None, 'Could not find matched lifted action for {}'.format(action_str)

  # 2. apply effect to the state
  var_obj_dict_action = matched_action.translate_objects_list_to_dict(action_object)
  new_state, _cost = matched_action.apply_effect(state, var_obj_dict_action)

  return new_state

def check_precondition(env_dictionary, operator_str, state, debug=False):
  ''' This function checks if the state is valid for precondition of the operator_str
  inputs:
  - env: a dictionary of HDDL components
  - operator_str: string of operator name and objects
  - state: list of predicates
  output:
  - boolean of whether the state is valid for precondition
  '''
  # 1. look up lifted operator:
  operator_name, operator_object = extract_object_list(operator_str)
  matched_oper = None
  for lifted_method in env_dictionary['lifted methods list']:
    if lifted_method.name == operator_name:
      matched_oper = lifted_method
      break
  if matched_oper == None:
    for lifted_action in env_dictionary['lifted actions list']:
      if lifted_action.name == operator_name:
        matched_oper = lifted_action
        break
  assert matched_oper != None, "Could not find lifted method or action matching operator {}".format(operator_str)

  # 2. check precondition with the state:
  var_object_dict = matched_oper.translate_objects_list_to_dict(operator_object)
  valid = matched_oper.precondition.check_precondition(state, var_object_dict, debug=debug)

  return valid


def check_collaborative_status(env_dictionary, agent, operator_str):
  '''
  This function check collaborative status of the operator string
  input:
  - env: env dictionary
  - agent: Agent instance of the current agent doing the operator
  - operator_str: string of operator
  output: tuple of:
  - collaborative status (boolean)
  - collab_agent_name (list of string)
  '''
  collaborative = False
  collab_agent_name = []
  for other_agent_name in env_dictionary['type object dict']['agent']:
    if agent.name != other_agent_name and other_agent_name in operator_str:
      collaborative = True
      collab_agent_name.append(other_agent_name)

  return collaborative, collab_agent_name

def get_invalid_by_order_subtasks(completed_subtasks, ordering_list, checking_subtasks_list):
  '''
  Return a list of invalid subtasks by order rules
  inputs:
  - completed_subtasks: list of just completed grounded subtasks
  - ordering_list: list or grounded ordered tuples
  - checking_subtasks_list: list of subtasks we want to check if valid or not
  output:
  - invalid_by_other: list of invalid subtasks
  '''
  invalid_by_order = set()
  completed_subtasks_not_relevant = True
  prior_subtask_dictionary = generate_prior_subtask_dictionary(ordering_list)
  for completed_subtask in completed_subtasks:
    if completed_subtask in prior_subtask_dictionary.keys():
      # add the list of subtasks that prior to the completed subtask in terms of ordering
      invalid_by_order.update(prior_subtask_dictionary[completed_subtask])
      completed_subtasks_not_relevant = False

  if completed_subtasks_not_relevant: #invalid_by_order contains tho has later order
    for subtask in checking_subtasks_list:
      if subtask in prior_subtask_dictionary.keys():
        for other_sub in checking_subtasks_list:
          if other_sub!=subtask and other_sub in prior_subtask_dictionary[subtask]:
            invalid_by_order.add(subtask)
    return invalid_by_order

  completed_set = invalid_by_order.union(completed_subtasks)
  for subtask in checking_subtasks_list:
    if subtask not in invalid_by_order and subtask in prior_subtask_dictionary.keys():
      if not set(prior_subtask_dictionary[subtask]).issubset(completed_set):
        # any subtask that has prior list not being a subset of completed set, is invalid
        invalid_by_order.add(subtask)

  return invalid_by_order

def generate_prior_subtask_dictionary(ordering_list):
  '''
  '''
  prior_subtask_dictionary = dict()
  for (t_before, t_after) in ordering_list:
    if t_after not in prior_subtask_dictionary.keys():
      prior_subtask_dictionary[t_after] = [t_before]
    else:
      prior_subtask_dictionary[t_after].append(t_before)
    if t_before in prior_subtask_dictionary:
      prior_subtask_dictionary[t_after] += prior_subtask_dictionary[t_before]
    for k in prior_subtask_dictionary.keys():
      if t_after in prior_subtask_dictionary[k] and t_before not in prior_subtask_dictionary[k]:
        prior_subtask_dictionary[k].append(t_before)
      if t_before in prior_subtask_dictionary.keys():
        prior_subtask_dictionary[k] = list(set(prior_subtask_dictionary[t_before]+prior_subtask_dictionary[k]))


  return prior_subtask_dictionary

#unit test:
# ordering_list = [('1','2'),('2','3'),('3','5'),('4','5')]
# ordering_list = [('1','2'),('3','5'),('4','5'),('2','3'),('0','1')]


