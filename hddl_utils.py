import copy
import itertools
import numpy as np
import torch

###  PARSING HELPER FUNCTIONS  ###

def read_hddl(hddl_filename):
    """ Read HDDL file and return string
    """
    file_str = open(hddl_filename, 'r')
    return str(file_str.read())

def split_components(hddl_str, headers, split_character = '(:'):
  ''' Split the components of hddl_str into different components
      This can be applied to
          domain, problem - split_character is '(:'
          method, action - split_character is ':'

  input: hddl_str, headers, split_character
  output: dictionary of {header: list of components}
  '''
  components = list(hddl_str.split(split_character))
  com_dict = {h:[] for h in headers}
  for c in components:
    for header in headers:
      if header in c[0:len(header)]:
        com_dict[header].append(c)
        continue

  return com_dict

def clean_str(org_str):
  '''remmove unnecesscary characters (whitespaces, extra parentheses)
  '''
  cleaned_str = copy.copy(org_str)
  cleaned_str = cleaned_str.replace('\n','')
  cleaned_str = cleaned_str.replace('\t', ' ')

  # remove extra whitespace
  while '  ' in cleaned_str:
    cleaned_str = cleaned_str.replace('  ',' ')
  cleaned_str = cleaned_str.replace(') )','))')
  cleaned_str = cleaned_str.replace('( )', '()')

  # remove exceed ')':
  extra = cleaned_str.count(')') - cleaned_str.count('(')
  i = 0
  last_ind = -1
  while extra > 0 and i < len(cleaned_str):
    if cleaned_str[-i-1] == ')':
      last_ind = -i-1
      extra -= 1
    i+=1
  cleaned_str = cleaned_str[0:last_ind]

  return cleaned_str

def extract_full_str(segment_str):
  '''
  this function extract complete str by parentheseses from segment_str
  input:
  - segment_str
  return:
  - the first full str starting from the first '('
  '''
  open_paren_list = []
  close_paren_list = []
  i = 0
  while len(open_paren_list) == 0 or (len(open_paren_list)!=len(close_paren_list) and i < len(segment_str)):
    ch = segment_str[i]
    if ch == '(':
      open_paren_list.append(i)
    elif ch == ')':
      close_paren_list.append(i)
    i+=1

  if len(open_paren_list)!=len(close_paren_list) or len(open_paren_list) == 0:
    return segment_str
  else:
    return segment_str[open_paren_list[0]:close_paren_list[-1]+1]


def enumerate_state(state, grounded_predicate_list):
  '''
  This function return one-hot list of state's predicates in the grounded_predicate_list
  '''
  num_state = []
  for pre in grounded_predicate_list:
    if pre in state:
      num_state.append(1)
    else:
      num_state.append(0)
  return num_state

def enumerate_list(target, reference_list, exist=1, non_exist = 0, lifted = False):
  '''Convert target list into matrix of len reference_list:
  input:
  - target: list of string
  - reference_list: list of strings, that must containt all element in target list
  - exist: value if the element of reference_list in target, default = 1
  - non_exist: value if the element of reference_list NOT in target, default = 0
  output:
  -num_target: list with len = len(reference_list), with value {exist, non_exist}
  '''
  num_target = []
  for ele in reference_list:
    if ele in target:
      num_target.append(exist)
    else:
      num_target.append(non_exist)
  return num_target

def find_corresponding_task(env_dictionary, method_str):
  '''
  Find the corresponding task of the method
  inputs:
  - env_dictionary: HDDLEnv instance
  - method_str: string of the grounded method
  output: 
  - co-task: corresponding task (string)
  '''
  method_name, method_obj = extract_object_list(method_str)
  matched_method = None
  for m in env_dictionary['lifted methods list']:
    if m.name == method_name:
      matched_method = m
      break
  assert m != None, "ERROR: couldn't find lifted method for {}".format(method_str)
  co_task = matched_method.print_corresponding_grounded_task(method_obj)
  return co_task

def parse_parameters(para_str):
  '''Interpret parameters string into parameters_type_list and parameters_dict {var_name: var_type}
  input: para_str can be as "parameters (?x - type1 ...)" or "?x - type1..."
  output:
  - parameters_name_list: [?var1, ?var2,...]
  - parameters_type_list: [type1, type2,...]
  - parameters_dict: {?var1: type1, ?var2: type2,...}
  '''
  parameters_dict = dict()
  parameters_type_list =[]
  parameters_name_list = []
  if '(' in para_str:
    params = para_str.split('(')[1]
    params = params.replace(')','')
    params = params.replace('\n', '')
    params = params.replace ('\t','')
  else:
    params = para_str
  params_com = params.split(' ')
  undefined_var = []
  next_is_type = False
  for c in params_com:
    if c == '':
      continue
    elif c == '-':
      next_is_type = True
    elif '?' in c:
      undefined_var.append(c)
      parameters_name_list.append(c)
    elif next_is_type:
      next_is_type = False
      for v in undefined_var:
        parameters_dict[v] = c
        parameters_type_list.append(c)
      undefined_var = []
    else:
      assert False, 'Wrong format for Parameters of operator, particularly: \n parameter str is {} \n and space-separated components is {}'.format(para_str, params_com)
      # print('Wrong format for Parameters of action, parameter str and parameter components:', (para_str, params_com))
  return parameters_name_list, parameters_type_list, parameters_dict

def enumerate_parameters(parameters_type_list, types_list):
  '''
  generate parameters_list: list of indices of types of parameters
  use information from parameters_type_list and types_list
  # outdated (?)
  '''
  parameters_id_list = []
  for vt in parameters_type_list:
    parameters_id_list.append(types_list.index(vt))
  return parameters_id_list

def generate_lifted_predicate(predicates_str, debug=False):
  ''' Interpret predicate string from domain and generate a list of lifted predicate
  inputs:
  - predicates_str: 'predicates \n (pred1 ?x - type1 ?y - type2) ...'
  - debug: debug flag
  output:
  - list of instances of lifted predicates
  '''
  gen_pre_list = []
  pre_list = predicates_str.split('(')[1:]
  pre_list_2 = []
  for pre in pre_list:
    pre = pre.split(')')[0]
    gen_pre_list.append(LiftedPredicate(pre,debug))
  return gen_pre_list

def generate_object_dict(types_str, objects_str, constant_str=''):
  '''Generate dictionaries for objects/constants and type hierarchy in HDDL.

    Inputs:
        types_str (str): The types section from an HDDL domain file.
        objects_str (str): The objects section from an HDDL problem file.
        constant_str (str, optional): The constants section from an HDDL domain file. Default is an empty string.

    Outputs:
        tuple:
            - type_object_dict: {type: list of all objects and constants in the type}.
            - supertype_type_dict: {supertype: list of all relevant types}.
  '''
  # 1. generate type_dict using info from types_str
  types_str = types_str[len('types')+1:]
  types_str = types_str.replace('\t',' ')
  types_lines = types_str.split('\n')
  type_dict = {}
  next_is_supertype = False
  for a in types_lines:
    a_com = a.split(' ')
    undefined_type_list = []
    for a_c in a_com:
      if a_c == '':
        continue
      elif a_c == '-':
        next_is_supertype = True
      elif next_is_supertype:
        next_is_supertype = False
        if a_c in type_dict.keys():
          type_dict[a_c] += undefined_type_list
        else:
          type_dict[a_c] = undefined_type_list
        undefined_type_list = []
      else:
        undefined_type_list.append(a_c)
  # make sure all values are root types (not super type):
  def check_valid_type_dict(type_d):
    value_list = []
    for v in type_dict.values():
      value_list += v
    for sup in type_dict.keys():
      if sup in value_list:
        return False
    return True

  valid_type_dict = False
  while not check_valid_type_dict(type_dict):
    for sup1 in type_dict.keys():
      for t in type_dict[sup1]:
        if t in type_dict.keys():
          type_dict[sup1].remove(t)
          type_dict[sup1] += type_dict[t]
          type_dict[sup1] = list(set(type_dict[sup1])) #remove duplicate types

  #2. Generate object_dict based on objects_str and type_dict
  objects_str = objects_str[len('objects'):]
  objects_str = objects_str.replace('\n','')
  objects_str = objects_str.replace(')','')
  objects_str = objects_str.replace('\t', ' ')
  com_list = objects_str.split(' ')
  if len(constant_str) != 0:
    constant_str = constant_str[len('constants'):]
    constant_str = constant_str.replace('\n', '')
    constant_str = constant_str.replace(')','')
    constant_str = constant_str.replace('\t',' ')
    com_list += constant_str.split(' ')
  
  next_is_type = False
  all_keys = set(type_dict.keys())
  for a in type_dict.values():
    all_keys.update(set(a))
  object_dict = {}
  undefined_object = []
  for c in com_list:
    if c == '':
      continue
    elif c == '-':
      next_is_type = True
    elif next_is_type:
      next_is_type = False
      if c not in object_dict.keys():
        object_dict[c] = undefined_object
      else:
        object_dict[c] += undefined_object
      undefined_object = []
    else:
      undefined_object.append(c)
  #supertype may not added to the object_dict, make sure to do so:
  for sup in type_dict.keys():
    if sup not in object_dict.keys():
      v = []
      for t in list(type_dict[sup]):
        if t in object_dict.keys():
          v += object_dict[t]
      object_dict[sup] = list(v)

  return object_dict, type_dict

def enumerate_string(input_str, header_list, var_list, var_internal_dict = None):
    '''This function convert string into a list of indices of each element in their corresponding list
    inputs:
    - input_str: a string that needs to be convert, e.g. 'at truck-0 city-0'
    - header_list: list of lifted type of the string, e.g. lifted_predicate_list
    - var_list: list of variables, which can be types (lifted) or objects (grounded)
    - var_internal_dict: dictionary to convert var_name into corresponding object, e.g. {'?v1': 'truck-0'},
                        which is needed if '?' in the string
    '''
    num_list = []
    input_com = input_str.split(' ')
    name = input_com[0]
    for i, h in enumerate(header_list):
        if h.name == name:
            num_list.append(i)
    assert len(num_list) == 1 #if not, input_str doesn't belong to header_list
    for param in input_com[1:]:
        if '?' in param:
            param_index = var_list.index(var_internal_dict[param])
        else:
            param_index = var_list.index(param)
        num_list.append(param_index)

    return num_list



def convert_grounded_to_lifted_index(grounded_list, lifted_list, object_list):
  '''
  This function convert grounded predicates/operators into indices of lifted_list and object_list
  inputs:
  -grounded_list: list of grounded predicates or operators
  - lifted_list: list of lifted predicates of operators accordingly
  - object_list: list of all objects in the environment
  outpus:
  - lifted_indices: list of indices of lifted mentioned in grounded
  - object_indices: lsit of indices of objects mentioned in grounded
  '''
  lifted_indices = list()
  object_indices = list()
  grounded_str = ' \n '.join(grounded_list)
  grounded_name_set = set()
  object_set = set()
  for grounded_element in grounded_list:
    name, obj = extract_object_list(grounded_element)
    grounded_name_set.add(name)
    object_set.update(obj)

  for ind_lifted, lifted in enumerate(lifted_list):
    if lifted.name in grounded_name_set:
      lifted_indices.append(ind_lifted)

  for ind_obj, obj in enumerate(object_list):
    if obj in object_set:
      object_indices.append(ind_obj)

  return lifted_indices, object_indices



def enumerate_index_to_binary_matrix(indices_list, array_len, exist=1, non_exist=0, device='cpu'):
    '''
    Generate a binary tensor with len of array_len, and value with index in indices_list be `exist` (default to 1),
    other values are `non_exist` (default to 0)
    inputs:
    - indices_list: list of indices
    - array_len: len of the output array
    - exist: value of the array when its index in indices_list
    - non_exist: value of the array when its index not in indices_list
    - device: device to place the tensor on (CPU or GPU)
    output:
    - binary_tensor: binary PyTorch tensor on the specified device
    '''
    
    # Handle case where indices_list is empty
    if indices_list:
        assert max(indices_list) < array_len, "The max index in the indices_list, index {} is out of array with length {}".format(max(indices_list), array_len)

    # Initialize the binary tensor with `non_exist` values
    binary_tensor = torch.full((array_len,), non_exist, dtype=torch.float32, device=device)

    # Set the values at indices in indices_list to `exist`
    if indices_list:
        binary_tensor[indices_list] = exist

    return binary_tensor


def parse_clause(clause_str):
    '''Parse clause into list like:
        ['and',pred1_str, pred2_str, ...]
        or ['or', ['and',pred1, p2,...], ['not',p3],...],
        or ['when',pre1, pre2] (equivalent to ['or', ['not',pre1], pre2])
    TODO:
    1. indices of '(', ')'
    2. handling 'and', 'or', 'not', 'when'
    3. convert precondition into list of string: ['and',pred1_str, pred2_str, ...]
    or ['or', ['and',pred1, p2,...], ['not',p3],...], or ['when',pre1, pre2] (equivalent to ['or', ['not',pre1], pre2])

    '''
    precondition_str = clean_str(clause_str)
    output_list = []
    indices_open = []
    indices_close = []
    for i, ch in precondition_str:
        if ch == '(':
            indices_open.append(i)
        elif ch == ')':
            indices_close.append(i)
    #remove extra ')':
    # note that number of close parenthesis must always >= number of open parenthesis
    assert len(indices_close) >= len(indices_open), "Number of close parenthesis is smaller than close parenthesis!"
    if len(indices_close) > len(indices_open):
        indices_close = indices_close[0:len(indices_open)]

    open_close_dict = dict()
    open_close_tuples = []
    for close_i in indices_close:
        best_o = None
        for open_i in indices_open:
            if open_i not in open_close_dict.keys():
                if open_i < close_i:
                    best_o = open_i
        # assign to dict:
        open_close_dict[best_o] = close_i
        open_close_tuples.append((best_o,close_i))

    def list_str(s):
        '''convert string s into list as described in 3
        '''
        lis_str = []
        special_words = ['and', 'or', 'when', 'not']
        if lis_str.split(' ')[0] in special_words:
            lis_str.append(lis_str.split(' ')[0])

    pre_com = precondition_str[indices_open[0]:open_close_dict[indices_open[0]]].split(' ')
    conj_clauses = []
    disj_clauses = []
    conditional_clauses = []
    negation_clauses = []
    atoms = []
    for open_i in open_close_dict.keys():
        if precondition_str[open_i+1 : open_close_dict[open_i]].split(' ')[0] == 'and':
            conj_clauses.append(tuple(open_i+5,open_close_dict[open_i]-1))
        elif precondition_str[open_i+1 : open_close_dict[open_i]].split(' ')[0] == 'or':
            disj_clauses.append(tuple(open_i+4 ,open_close_dict[open_i]-1))
        elif precondition_str[open_i+1 : open_close_dict[open_i]].split(' ')[0] == 'when':
            conditional_clauses.append(tuple(open_i+6,open_close_dict[open_i]-1))
        elif precondition_str[open_i+1:open_close_dict[open_i]].split(' ')[0] == 'not':
            negation_clauses.append(tuple(open_i+5,open_close_dict[open_i]-1))
        else:
            atoms.append(tuple(open_i+1, open_close_dict[open_i]-1))

    #looking for the subclauses in between a pair of parenthesis:
    and_subclauses = dict()
    for o, c in conj_clauses:
        and_subclauses[(o,c)] = []
        for o_i in open_close_dict.keys():
            if o_i >= o and open_close_dict[o_i] <= c:
                and_subclauses[(o,c)].append((o_i,open_close_dict[o_i]))


def extract_object_list(grounded_str):
  '''convert the grounded_str to (name, object_list)
  '''
  if grounded_str == None:
    return 'None', []
  com = grounded_str.split(' ')
  name = com[0]
  object_list = com[1:]
  return name, object_list


def check_subtask_with_world_state(env_dictionary, world_state, incompleted_grounded_subtasks):
  ''' This function checks if any tasks/actions in the incompleted_grounded_subtasks are in fact archieved
  input:
  - env_dictionary: env dict
  - world_state: list of predicates represent the world state
  - incompleted_grounded_subtasks: list of grounded tasks or actions to check
  output:
  - list of actions or tasks that are checked to be completed

  '''
  completed_subtasks = []
  for subtask in incompleted_grounded_subtasks:
    # find matched task or action
    subtask_name, subtask_obj = extract_object_list(subtask)
    matched_subtask = None
    for t in env_dictionary['lifted tasks list']+env_dictionary['lifted actions list']:
      if t.name == subtask_name:
        matched_subtask = t
        break
    # if matched_subtask != None:
    #   for a in env_dictionary['lifted actions list']:
    #     if a.name == subtask_name:
    #       matched_subtask = a
    #       break

    assert matched_subtask!= None, "Cannot find matched task or action for the subtask {}".format(subtask)
    subtask_var_obj_dict = matched_subtask.translate_objects_list_to_dict(subtask_obj)
    # check effect:
    new_world_state = matched_subtask.effect.apply_effect(world_state,subtask_var_obj_dict)
    if set(new_world_state) == set(world_state):
      completed_subtasks.append(subtask)

  return completed_subtasks


def extract_dynamic_predicates(lifted_predicates_list, list_effect_str):
  '''
  This function return the predicates that are used in the effect
  inputs:
  - lifted_predicates_list
  - list_effect_str: list of strings of effect
  outputs:
  - set of indices of lifted predicates that are mentioned in the effect str
  '''
  dynamic_predicate_set = set()
  for i,lifted_pre in enumerate(lifted_predicates_list):
    head_word_pre = lifted_pre.name
    head_phrase = '(' + head_word_pre + ' '
    if head_phrase in '\n '.join(list_effect_str):
      dynamic_predicate_set.add(i)
  return dynamic_predicate_set


def check_collaborative_status(env_dictionary, agent, operator_str):
  '''
  This function check collaborative status of the operator string
  input:
  - env_dictionary: env dictionary
  - agent: Agent instance of the current agent doing the operator
  - operator_str: string of operator
  '''
  collaborative = False
  collab_agent_name = []
  for a in env_dictionary['type object dict']['agent']:
    if agent.name != a and a in operator_str:
      collaborative = True
      collab_agent_name.append(a)

  return collaborative, collab_agent_name

def extract_full_str(segment_str):
  '''
  this function extract complete str by parentheseses from segment_str
  input:
  - segment_str
  return:
  - the first full str starting from the first '('
  '''
  open_paren_list = []
  close_paren_list = []
  i = 0
  while len(open_paren_list) == 0 or (len(open_paren_list)!=len(close_paren_list) and i < len(segment_str)):
    ch = segment_str[i]
    if ch == '(':
      open_paren_list.append(i)
    elif ch == ')':
      close_paren_list.append(i)
    i+=1

  if len(open_paren_list)!=len(close_paren_list) or len(open_paren_list) == 0:
    return segment_str
  else:
    return segment_str[open_paren_list[0]:close_paren_list[-1]+1]


def extract_dynamic_predicates(lifted_predicates_list, list_effect_str):
  '''
  This function return the predicates that are used in the effect
  inputs:
  - lifted_predicates_list
  - list_effect_str: list of strings of effect
  outputs:
  - set of indices of lifted predicates that are mentioned in the effect str
  '''
  dynamic_predicate_set = set()
  for i,lifted_pre in enumerate(lifted_predicates_list):
    head_word_pre = lifted_pre.name
    head_phrase = '(' + head_word_pre + ' '
    if head_phrase in '\n '.join(list_effect_str):
      dynamic_predicate_set.add(i)
  return dynamic_predicate_set


#################################
### CLASSES  ###

class LiftedPredicate:
  def __init__(self,predicate_str, debug=False):
    self.lifted_str = predicate_str
    self.name = predicate_str.split(' ')[0].replace('\n','')
    self.num_var = predicate_str.count('?')
    self.ordered_var_name, self.var_dict = self.generate_var_dict()
    if debug:
      print(self.var_dict)

    self.num_grounded = 0 #number of possible grounded predicates, initially set to 0

  def generate_var_dict(self):
    ''' generate dictionary of variable: {var_name: var_type}
        e.g. {'?v1': 'vehicle}
    '''
    vars_com = self.lifted_str.split(' ')[1:]
    ordered_var_name = []
    var_dict = {}
    untyped_var = []
    next_is_type = False
    for com in vars_com:
      if '?' in com:
        ordered_var_name.append(com)
        untyped_var.append(com)
      elif com == '-':
        next_is_type = True
      elif next_is_type:
        for v in untyped_var:
          var_dict[v] = com
        next_is_type = False
        untyped_var = []
    return ordered_var_name, var_dict

  def __str__(self):
    return self.lifted_str

  def __eq__(self, other):
    return self.lifted_str == other.lifted_str

  def print_grounded_predicate(self, object_list):
    ans = self.name
    for i in object_list:
      ans = ans + ' ' + str(i)
    return ans

  def generate_grounded_predicate(self, objects_dict):
    ''' Generate a list of all possible grounded predicates based on the list of objects
    inputs: objects_dict: dictionary of types and objects, e.g. {'vehicle': ['truck-0','truck-1']}
    outputs: list of strings of grounded predicates in alphabetical order, e.g.:
              ['at truck-0 l0','at truck-0 l1', 'at truck-1 l0',...]
    '''
    var_options_list = list(objects_dict[self.var_dict[v_n]] for v_n in self.ordered_var_name)
    # print('var options list: ',var_options_list)
    var_combination_list = list(itertools.product(*var_options_list))
    # print('var_com_list:', var_combination_list)
    self.num_grounded = len(var_combination_list)
    grounded_predicate_list = []
    for l in var_combination_list:
      grounded_predicate_list.append(self.print_grounded_predicate(l))
    self.grounded_predicate_list = sorted(grounded_predicate_list)
    return sorted(grounded_predicate_list)

############

class Task:
  ''' Task class is init with task_str: 'task task_name ?var_name1 - var_type1 ...'
  components:
  - name
  - var_name_list: ['?v', '?l1', '?l2']
  - var_type_list: must have same length as var_name, e.g. ['vehile','location','location']
  - related_methods_list: list of related lifted method instances

  methods:
  - analyze_task_str
  - __str__: print the task string
  - print_grounded_task: print the string of grounded task, e.g. 'drive truck-0 city-0 city-1'


  '''
  def __init__(self,task_str):
    # Note: task_str has format: 'task <task_name> ?<var_name1> - <var_type1>...'
    self.task_str = task_str
    self.name = task_str.split('\n')[0].split(' ')[1].replace('\n','')
    name_space = ' '+self.name + ''
    # print("name_space of task_str {} is {}".format(task_str,name_space))
    idx_name = task_str.index(name_space)
    self.task_headers = ['task','parameters', 'effect']
    self.com_dict = split_components(self.task_str, self.task_headers, split_character=':')
    self.parameters_name_list, self.parameters_type_list, self.parameters_dict = parse_parameters(self.com_dict['parameters'][0])

    # self.parameters_str = task_str[idx_name+len(name_space):]
    # self.parameters_name_list, self.parameters_type_list, self.parameters_dict = parse_parameters(self.parameters_str)
    self.related_methods_list = []
    self.is_collaborative = False # make sure to check if task is collaborative by comparing parameters_name_list with supertype-type dict
    self.collaborative_agents = 0
    self.effect = Effect(clean_str(self.com_dict['effect'][0]))

  def check_collaborative(self, supertype_type_dict):
    ''' check collaborative characteristic of the task
    if the task requires collaboration, return True and number of agents needed
    else: return False and number of agent involved (usually 1)
    '''
    agent_count = self.parameters_type_list.count('agent')
    for t in self.parameters_type_list:
      for st in supertype_type_dict.keys():
        if st == t and 'agent' in supertype_type_dict[st]:
          agent_count += 1
    if agent_count >1:
      return True, agent_count
    else:
      return False, agent_count

  def print_grounded_task(self,var_object_dict):
    '''
    print grounded task string
    inputs:
    - var_object_dict: dictionary of {variable:object}, e.g. {'v':'truck-0'}
    output:
    - grounded_task_str: string of grounded task
    '''
    grounded_task_str = copy.copy(self.task_str)
    for var in var_object_dict.keys():
      grounded_task_str.replace(var, var_object_dict[var])

    assert '?' in grounded_task_str, 'not completely grounded yet! {}'.format(grounded_task_str)
    return grounded_task_str

  def print_name(self, object_list = None, var_object_dict=None):
    '''
    print string of task
    inputs:
    - object_list: list of object, in the order of variables appear in :parameters of the task
    - var_object_dict: dictionary of {variable: object}, e.g. {'v':'truck-0'}
    outputs:
    - return task string, including task_name + objects (or variables)
    '''
    if object_list != None:
      s = self.name + ' '+' '.join(object_list)
    elif var_object_dict != None:
      param = []
      for v in self.parameters_name_list:
        param.append(var_object_dict[v])
      s = self.name + ' ' + ' '.join(param)
    else:
      s = self.name + ' ' + ' '.join(self.parameters_name_list)
    return s

  def generate_grounded_tasks(self,type_object_dict):
    ''' generate list of grounded actions with name and parameters
    input:
    - type_object_dict: dictionary of {type:object}, e.g. {'vehicle':['truck-0','car-0','truck-1']}
    output:
    - sorted list of grounded tasks
    '''
    var_options_list = list(type_object_dict[t] for t in self.parameters_type_list)
    var_combination_list = list(itertools.product(*var_options_list))
    self.num_grounded = len(var_combination_list)
    grounded_tasks_list = []
    for l in var_combination_list:
      grounded_tasks_list.append(self.print_name(object_list=l))
    self.grounded_actions_list = sorted(grounded_tasks_list)
    return sorted(grounded_tasks_list)

  def translate_objects_list_to_dict(self, objects_list):
    '''translate the objects_list of task into dictiornary {var_method:object}
    '''
    assert len(self.parameters_name_list) == len(objects_list),\
     "number of objects doens't match number of task var for task {}!\n parameter_name_list {} \n object_list{}".format(self.name, self.parameters_name_list, objects_list)
    ans = dict()
    for i,p in enumerate(self.parameters_name_list):
      ans[p] = objects_list[i]

    return ans

  def __str__(self):
    return self.task_str

  def __eq__(self, other):
    if other == None:
      return False
    return self.name == other.name

##############

class Method:
  '''
  components:
  - name, com_dict (dictionary of components), method_headers,
  - parameters_type_list, parameters_name_list, parameters_dict, parameters_id_list (index of parameter in the types list)
  - subtasks (list of subtasks), subtasks_ids (id of subtasks, for ordering), ordering
  - precondition (Precondition instance, has check_preconditon, ground_precondtion)
  - effect: Effect instance, has apply_effect for grounded method

  NOTE: Method stands along is lifted, to ground it, need var_object_dict (or object list that each element is compatible to method.parameter_name_list)
  e.g. Lifted Method: drive [?v, ?l1, ?l2] => grounded method: drive [truck-0, city-0, city-1]

  functions:
  - parse_subtask: dissect subtask string into list of subtasks and ordering
  '''
  def __init__(self,method_str, types_list, task_list = None):
    '''inputs:
    - method_str: string describing method: 'method <name> ...'
    - types_list: list of types existing in the domain
    - task_list: list of instances of Task that have been initiated
    '''
    self.method_str = method_str
    self.name = method_str.split('\n')[0].split(' ')[1].replace('\n','')
    self.method_headers = ['method','parameters', 'precondition','subtasks', 'ordered-subtasks', 'ordering', 'task']
    self.com_dict = split_components(self.method_str, self.method_headers, split_character=':')
    # print(self.com_dict)
    self.parameters_type_list = []
    self.parameters_dict = dict()
    self.parameters_name_list, self.parameters_type_list, self.parameters_dict = parse_parameters(self.com_dict['parameters'][0])
    self.parameters_id_list = enumerate_parameters(self.parameters_type_list, types_list)
    self.num_grounded = 0
    self.subtasks = []
    self.subtasks_ids = []
    self.ordering = []
    self.parse_subtask()
    if ':precondition' in method_str:
      self.precondition = Precondition(clean_str(self.com_dict['precondition'][0]))
    else:
      self.precondition = Precondition(None)
    self.grounded_methods_list = []
    self.task_name = self.com_dict['task'][0].split('task (')[1].split(' ')[0]
    self.task_parameters_name_list = self.com_dict['task'][0].split('task (')[1].split(')')[0].split(' ')[1:]
    self.task_str = self.task_name + ' ' + ' '.join(self.task_parameters_name_list)

    if task_list != None:
      # match with lifted task instance:
      for task in task_list:
        if task.name == self.task_name:
          task.related_methods_list.append(self)

  def __str__(self):
    return self.method_str

  def __eq__(self, other):
    if isinstance(other, Method):
      return self.name == other.name
    else:
      return False

  def print_name(self, object_list = None, var_object_dict=None):
    '''
    '''
    if object_list != None:
      s = self.name + ' '+' '.join(object_list)
    elif var_object_dict != None:
      param = []
      for v in self.parameters_name_list:
        param.append(var_object_dict[v])
      s = self.name + ' ' + ' '.join(param)
    else:
      s = self.name + ' ' + ' '.join(self.parameters_name_list)
    return s

  def print_corresponding_grounded_task(self, object_list = None, var_object_dict = None):
    '''
    print the corresponding grounded task
    input:
    - object_list: list of object name, in order similar to parameters', default to None
    - var_object_dict: dictionary of var: object, default to None
    - if both are None, print out the lifted task of the method
    output:
    - string of grounded task of the method
    '''
    s = copy.copy(self.task_str)
    if object_list != None:
      for i, var in enumerate(self.parameters_name_list):
        s = s.replace(var, object_list[i])
    elif var_object_dict != None:
      for i, var in enumerate(self.parameters_name_list):
        s = s.replace(var, var_object_dict[var])
    return s

  def parse_subtask(self):
    ''' parsing subtask
    considering cases: (1) ordered-subtasks, (2) subtasks + ordering, (3) subtasks (no ordering)
    => read and return: list of subtasks, and list of ordering logics in pair e.g. [(t1,t2), (t2,t3), (t3,t4)]
    '''
    # print("parsing subtasks for method: ",self.name)
    # print('self.com_dict of method', self.com_dict)
    if len(self.com_dict['ordered-subtasks']) > 0:
      # self.com_dict['subtasks'] can be "ordered-subtasks (and (subtask-1)\n(subtask-2)...)" OR "ordered-subtasks (and)" OR "ordered-subtasks ()"
      # ordered-subtasks appear in format of conjuntive list of subtasks or a single subtask, no need subtask-id
      ordered_subtasks_str = self.com_dict['ordered-subtasks'][0]
      ordered_subtasks_str = clean_str(ordered_subtasks_str.split('ordered-subtasks (')[1])
      #use Literal to analyze the string:
      subtask_literal = Literal(ordered_subtasks_str)
      if subtask_literal.is_empty:
        self.subtasks = []
      elif subtask_literal.is_atom:
        self.subtasks.append(subtask_literal.literal_str)
      elif subtask_literal.is_and:
        for j, child in enumerate(subtask_literal.child_clauses):
          assert child.is_atom, 'check ordered-subtasks format of method {}!!!'.format(self.name)
          self.subtasks.append(child.literal_str)
          if j>0:
              self.ordering.append((subtask_literal.child_clauses[j-1].literal_str, child.literal_str))
      else:
        print("ERROR: wrong format for ordered-subtasks", ordered_subtasks_str)

    elif len(self.com_dict['subtasks']) > 0:
      # self.com_dict['subtasks'] can be "subtasks (and (id1 (subtask-1))\n(id2 (subtask-2))...)" OR "subtasks (and)" OR "subtasks ()"
      subtasks_str = clean_str(self.com_dict['subtasks'][0].split('subtasks (')[1])
      subtask_literal = Literal(subtasks_str)
      if subtask_literal.is_atom and not subtask_literal.is_empty:
        self.subtasks.append(subtask_literal.literal_str)
      elif subtask_literal.is_and:
        for child in subtask_literal.child_clauses:
          assert child.is_atom, 'Check subtask format of method {}'.format(self.name)
          if '(' in child.literal_str:
            subtask_id = child.literal_str.split(' ')[0]
            subtask = child.literal_str.split('(')[1]
            subtask = subtask.split(')')[0]
          else:
            subtask_id = None
            subtask = child.literal_str
          self.subtasks_ids.append(subtask_id)
          self.subtasks.append(subtask)
      else:
        print("ERROR: wrong format for subtasks in method:", self.name)

      #handling ordering:
      if len(self.com_dict['ordering']) > 0:
        ordering_str = clean_str(self.com_dict['ordering'][0].split('ordering (')[1])
        ordering_literal = Literal(ordering_str)
        # Note that ordering_str should be in format of 'and (< task_id1 task_id2) (...)...'
        if ordering_literal.is_atom:
          self.ordering.append((ordering_literal.literal_str.split(' ')[1], ordering_literal.literal_str.split(' ')[2]))
        elif ordering_literal.is_and:
          for child in ordering_literal.child_clauses:
            t_id_before = child.literal_str.split(' ')[1]
            t_id_after = child.literal_str.split(' ')[2]
            t_before = self.subtasks[self.subtasks_ids.index(t_id_before)]
            t_after = self.subtasks[self.subtasks_ids.index(t_id_after)]
            self.ordering.append((t_before, t_after))


  def generate_grounded_methods(self, type_object_dict, task_var_object_dict = None):
    '''Generate all possible grounded methods
    inputs: 
    - type_object_dict: dictionary of {type: objects}, e.g. {'vehicle': ['truck-0','truck-1', 'car-0']}
    - task_var_object_dict: dictionary of task's variables and object, e.g. {'v':['truck-0','truck-1', 'car-0']}
    outputs:
    - sorted list of grounded methods 
    '''
    var_options_list = []
    if task_var_object_dict != None:
      for i, v in enumerate(self.parameters_name_list):
        if v in task_var_object_dict.keys():
          var_options_list.append([task_var_object_dict[v]])
        else:
          var_options_list.append(type_object_dict[self.parameters_type_list[i]])
    else:
      var_options_list = list(type_object_dict[t] for t in self.parameters_type_list)
    var_combination_list = list(itertools.product(*var_options_list))
    # print('var_com_list:', var_combination_list)
    self.num_grounded = len(var_combination_list)
    grounded_methods_list = []
    for l in var_combination_list:
      grounded_methods_list.append(self.print_name(object_list=l))
    self.grounded_methods_list = sorted(grounded_methods_list)
    return sorted(grounded_methods_list)

  def translate_objects_list_to_dict(self, task_objects_list):
    '''translate the objects_list of task into dictiornary {var_method:object}
    input:
    - task_objects_list: list of objects in the order of the task's parameters
    output:
    - var_object_dict: dictionary of {var: objects}
    '''
    ans = dict()
    if len(self.task_parameters_name_list) == len(task_objects_list):
      for i,p in enumerate(self.task_parameters_name_list):
        ans[p] = task_objects_list[i]
    else: #this translation is for the method_object list
      for i,p in enumerate(self.parameters_name_list):
        ans[p] = task_objects_list[i]

    return ans




###########

class Action:
  '''
  NOTE:  similar to Method, Action stands alone is lifted, to ground it, provide var_object_dict or an object list that is compatible to action.parameters_name_list
  components:
  - name, action_str, action_headers, com_dict
  - parameters_name_list, parameters_type_list, parameters_dict
  - preconditon: Precondition instance (has ground option and check precondition option)
  - effect: Effect instance (has apply_effect for grounded action)
  - cost: 1 for now


  '''
  def __init__(self,action_str):
    self.action_str = action_str
    self.name = action_str.split('\n')[0].split(' ')[1].replace('\n','')
    self.action_headers = ['action','parameters','precondition','effect','cost']
    self.com_dict = split_components(self.action_str, self.action_headers, split_character=':')
    self.parameters_name_list, self.parameters_type_list, self.parameters_dict = parse_parameters(self.com_dict['parameters'][0])
    self.precondition = Precondition(clean_str(self.com_dict['precondition'][0]))
    self.effect = Effect(clean_str(self.com_dict['effect'][0]))
    self.cost = 1
    # self.grounded_action_list = []

  def __str__(self):
    return self.action_str

  def print_name(self, object_list = None, var_object_dict = None):
    '''print out the name of grounded action: <name> <var1/object1> <var2/obj2>...
    e.g. move ?v ?l1 ?l2,  OR: move truck-0 city-0 city-1
    input:
    - var_object_dict if want to print name of grounded action
    '''
    if var_object_dict == None and object_list == None:
      s = self.name + ' ' + ' '.join(self.parameters_name_list)
    elif object_list != None:
      s = self.name + ' '+ ' '.join(object_list)
    else:
      param = []
      for v in self.parameters_name_list:
        param.append(var_object_dict[v])
      s = self.name + ' ' + ' '.join(param)

    return s

  def translate_objects_list_to_dict(self, objects_list):
    '''translate the objects_list of task into dictiornary {var_method:object}
    '''
    assert len(self.parameters_name_list) == len(objects_list), "number of objects doens't match number of task var!\n parameter_name_list {} \n object_list{}".format(self.parameters_name_list, objects_list)
    ans = dict()
    for i,p in enumerate(self.parameters_name_list):
      ans[p] = objects_list[i]

    return ans



  def generate_grounded_actions(self,type_object_dict):
    ''' generate list of grounded actions with name and parameters
    '''
    var_options_list = list(type_object_dict[t] for t in self.parameters_type_list)
    var_combination_list = list(itertools.product(*var_options_list))
    self.num_grounded = len(var_combination_list)
    grounded_actions_list = []
    for l in var_combination_list:
      grounded_actions_list.append(self.print_name(object_list=l))
    self.grounded_actions_list = sorted(grounded_actions_list)
    return sorted(grounded_actions_list)

  def check_precondition(self, state, var_object_dict, debug=False):
    '''Check precondition of a grounded action to see if it is a valid action at the given state
    inputs:
    - var_object_dict: list of index of variables mapping from the objects_list, len of grounded_var = len(self.parameters_dict.keys())
    - state: list of indices of grounded predicates of the state from the grounded_predicate_list

    output: bool True or False
    '''
    return self.precondition.check_precondition(state, var_object_dict, debug=debug)

  def apply_effect(self, state, var_object_dict, debug = False):
    '''
    apply effect of the grounded action to the state
    inputs:
    - var_object_dict
    - state: current state, list of all predicates
    outputs:
    - new_state
    - cost (default to be constant self.cost)
    '''
    if self.effect.no_effect:
      return state, self.cost
    new_state = copy.deepcopy(state)
    #1. Ground the effect str:
    effect_str = clean_str(copy.deepcopy(self.com_dict['effect'][0]))
    effect_idx = effect_str.index('effect (')
    grounded_str = clean_str(effect_str[effect_idx + len('effect ('):])
    for var in var_object_dict.keys():
      grounded_str = grounded_str.replace(var, var_object_dict[var])
    if debug:
      print("grounded_str in action.apply_effect: ", grounded_str)
    assert '?' not in grounded_str, "The effect literal has not been completely grounded! current str: {}".format(grounded_str)
    effect_literal = Literal(grounded_str)
    #2. apply effect:
    # Note that the effect literal should be either conjuctive clauses or a singe atom or a single NOT atom
    if effect_literal.is_empty:
      return new_state, self.cost
    elif effect_literal.is_atom:
      if effect_literal.literal_str not in new_state:
        new_state.append(effect_literal.literal_str)
        if debug:
          print("appending new literal str to the state: {}".format(effect_literal.literal_str))
    elif effect_literal.is_not:
      assert effect_literal.child_clauses[0].is_atom, "must simplify this clause, after NOT should be just an atom. Effect literal: {}".format(effect_literal)
      if effect_literal.child_clauses[0] in state:
        new_state.remove(effect_literal.child_clauses[0])
        if debug:
          print('remove literal str from the state: {}'.format(effect_literal.child_clauses[0]))
    elif effect_literal.is_and:
      for child in effect_literal.child_clauses:
        if child.is_atom and child.literal_str not in new_state:
          new_state.append(child.literal_str)
          if debug:
            print("appending new literal str to the state: {}".format(child.literal_str))
        elif child.is_not and child.child_clauses[0].literal_str in new_state:
          new_state.remove(child.child_clauses[0].literal_str)
          if debug:
            print('remove literal str from the state: {}'.format(child.child_clauses[0].literal_str))
        
    else:
      print("Effect literal has incorrect categories! effect str:", effect_literal.literal_str)

    if debug:
      print("new state after action {} is: {}".format(self.print_name(var_object_dict=var_object_dict), '\n'.join(new_state)))
      print("empty predicate in new_state:", '' in new_state)
    return new_state, self.cost



#######
class Effect:
  '''class Effect has following methods:
  - self.generate_predicate_lists(): generate list of all lifted predicates mentioned in the effect (including adding or removing predicates)
  - self.apply_effect(state, var_object_dict): ground the effect with var_object_dict, then apply grounded effect to the state and return new state

  ASSUMPTIONS:
  - no cost in effect
  - effect is in the simpliest form: which is a conjunctive list of positive or negation of predicates. 
  In another words, it is either and list of atoms and NOT atoms, or a single atom or a single NOT atom
  '''
  def __init__(self,effect_str, var_object_dict=None, ):
    self.effect_str = effect_str # format: "effect (...)"
    self.var_object_dict =  var_object_dict #for lifted action, this dict is None
    self.adding_lifted_predicate_list, self.removing_lifted_predicate_list = self.generate_predicate_lists()
    self.eff_lit_str = clean_str(effect_str)[len('effect ('):]
    test_empty = self.eff_lit_str.replace(')','')
    test_empty = test_empty.replace(' ','')
    test_empty = test_empty.replace('\n','')
    self.no_effect = test_empty == ''

  def generate_predicate_lists(self):
    '''generate list of all predicates mentioned in the effect
    outputs: tuple of
    - adding_lifted_predicate_list
    - removing_lifted_predicate_list
    '''
    effect_str = self.effect_str
    effect_str = clean_str(effect_str.replace('\n',''))
    eff_com = effect_str.split(' ')
    current_pre = []
    next_is_removing = False
    adding_lifted_predicate_list = []
    removing_lifted_predicate_list = []
    for c in eff_com:
      if len(c) == 0:
        continue
      elif c == '(not':
        next_is_removing = not next_is_removing
      elif '(' == c[0]:
        current_pre = [c.replace('(','')]
      elif '?' in c and ')' not in c:
        current_pre.append(c)
      elif ')' in c: #end of current pre
        current_pre.append(c.replace(')',''))
        if next_is_removing:
          removing_lifted_predicate_list.append(' '.join(current_pre))
          next_is_removing = False
        else:
          adding_lifted_predicate_list.append(' '.join(current_pre))
        current_pre = []
    return adding_lifted_predicate_list, removing_lifted_predicate_list

  def __str__(self):
    return self.effect_str

  def apply_effect(self,state, var_object_dict):
    '''Function applies effect to the current state
    inputs:
    - state: list of predicates
    - var_object_dict: (for grounding): {?variable: object}
    output:
    - new_state
    '''
    if self.no_effect:
      return state
    new_state = copy.copy(state)
    grounded_str = copy.copy(self.eff_lit_str)
    #1. replace var with object:
    for var in var_object_dict:
      grounded_str = grounded_str.replace(var, var_object_dict[var])
    assert '?' not in grounded_str, "Not replace all var with object, double check var_object_dict! grounded_str is {}".format(grounded_str)

    effect_literal = Literal(grounded_str)
    #2. start applying effect to the state
    # Note that effect literal is either and list of atoms and not literals, or a single atom or a single not literal
    if effect_literal.is_empty:
      return new_state
    elif effect_literal.is_atom:
      if effect_literal.literal_str not in new_state:
        new_state.append(effect_literal.literal_str)
    elif effect_literal.is_not:
      assert len(effect_literal.child_clauses) == 1, "Only 1 clause should be after NOT, there are {} clauses!".format(len(effect_literal.child_clauses))
      atom = effect_literal.child_clauses[0]
      assert atom.is_atom, "after NOT should just be atom, please convert the literal to the simplest form"
      if atom.literal_str in new_state:
        new_state.remove(atom.literal_str)

    elif effect_literal.is_and:
      for child in effect_literal.child_clauses:
        if child.is_atom:
          new_state.append(child.literal_str)
        elif child.is_not:
          atom = child.child_clauses[0]
          assert atom.is_atom, "after NOT should just be atom, please convert the literal to the simplest form: {}".format(atom.literal_str)
          if atom.literal_str in new_state:
            new_state.remove(atom.literal_str)
        else:
          assert "Format error: Each clauses of conjunctive clause should be either atom or negation of atom!"

    return new_state



class Precondition:
    '''class Precondition:
    - initialized with precondition string
    - important methods:
      - self.ground_precondition(var_object_dict): ground the precondition with var_object_dict
      - self.check_precondition(state, var_object_dict): ground the precondition then check the precondition with the state
    '''
    def __init__(self, precondition_str):
        '''
        precondition_str is a string start with 'precondition (...)'
        '''
        self.precondition_str = precondition_str
        if self.precondition_str == None or self.precondition_str == '':
          self.precondition_str = None
          return
        pre_lit_str = clean_str(precondition_str)[len('precondition ('):]
#         pre_lit_str = pre_lit_str[:pre_lit_str.rindex(')')]
        if pre_lit_str.replace(' ','') == '':
          self.precondition_str = None
        self.precondition_literal = Literal(pre_lit_str)
        self.grounded_precondition_literal = None

    def __str__(self):
      return self.preconditon_str

    def ground_precondition(self, var_object_dict):
        '''
        1. Grounding the precondition with the var-object dictionary
        Note: var_object_dict in the form: {'?v' : truck-0}
        2. Check grounded precondition with state
        '''
        if self.precondition_str == None:
          print("No precondition!")
          return
        #replace ?var by object with var_object_dict:
        self.grounded_precondition_str = copy.copy(self.precondition_str)
        for var in var_object_dict.keys():
          self.grounded_precondition_str = self.grounded_precondition_str.replace(var, var_object_dict[var])

        pre_lit_str = clean_str(self.grounded_precondition_str)[len('precondition ('):]
        self.grounded_precondition_literal = Literal(pre_lit_str)

    def check_precondition(self, state, var_object_dict, debug=False):
        '''Checking the precondition (only works for grounded precondition)
        '''
        if self.precondition_str == None:
          # print("No precondition!")
          return True
        self.ground_precondition(var_object_dict)
        return self.grounded_precondition_literal.check_literal(state,debug=debug)


class Literal:
    ''' class Literal:
    - initalized with literal string
    - self.analyze_lit(): analyze literal string, disect the literal string into smaller subsets of literals
    - self.check_literal(state): check if the literal is correct given the state
    '''
    def __init__(self,lit_str, adding = 1):
        ''' Note: lit_str is like 'and (...) (...)' or 'or (...) (...)' or 'not (...)' or 'no_parenthesis'
        '''
        self.literal_str = lit_str
        self.is_and = False
        self.is_or = False
        self.is_not = False
        self.is_when = False
        self.is_atom = True
        self.is_compare = False
        self.child_clauses = []
        self.analyze_lit()
        self.adding = adding
        self.is_empty = False

    def __str__(self):
      return self.literal_str

    def analyze_lit(self):
        '''Analyze literal:
        1. if it starts with and, or, not => update elements: all child clauses
        2. else: it is atom => is_atom = True, child_clauses = []
        '''
        first_word = self.literal_str.split(' ')[0]
        self.is_and = first_word == 'and'
        self.is_or = first_word == 'or'
        self.is_not = first_word == 'not'
        self.is_when = first_word == 'when'
        self.is_compare = first_word == '='

        if self.is_and or self.is_or or self.is_not or self.is_when:
            self.is_atom = False
            # a. find index of start and stop boundary of child clauses:
            if '(' not in self.literal_str:
              self.is_empty = True
              return
            start_open_i = self.literal_str.index('(')
            open_count = 1
            close_count = 0
            current_open_i = start_open_i
            start_end_clause = [] # list of type(start_index, end_index)
            for i, c in enumerate(self.literal_str[start_open_i+1:]):
                if c == '(':
                    if open_count == 0:
                        current_open_i = int(i + start_open_i+1)
                    open_count +=1
                elif c == ')':
                    close_count +=1
                if open_count == close_count and open_count != 0:
                    start_end_clause.append((current_open_i+1, i + start_open_i))
                    child_str = self.literal_str[int(current_open_i + 1): int(i + start_open_i+1)]
                    if child_str.replace(' ','') != '':
                      self.child_clauses.append(Literal(self.literal_str[int(current_open_i + 1): int(i + start_open_i+1)]))
                    open_count = 0
                    close_count = 0
        else:
            self.is_atom = True


    def check_literal(self, state, debug=False):
        '''Check if the literal is valid in the state
        1. if the literal is atom: check valid by comparing
        2. if the literal is AND: check each clause
        3. if the literal is OR: check each clause
        4. if the literal is When: check not clause-0 OR clause-1
        '''
        if self.is_empty:
          return True
        elif self.is_and:
            for element in self.child_clauses:
                if not element.check_literal(state, debug=debug):
                    if debug:
                      print("[line 1090] False because of not having {} in state".format(element.literal_str))
                      print("lit str:", self.literal_str)
                    return False
            return True
        elif self.is_or:
            for element in self.child_clauses:
                if element.check_literal(state, debug=debug):
                    return True
            return False
        elif self.is_not:
            return not self.child_clauses[0].check_literal(state, debug=debug)
        elif self.is_when:
            assert len(self.child_clauses) !=2, "Need 2 clauses after when, this literal {} has {} clauses".format(self.literal_str,len(self.child_clauses))
            return not self.child_clauses[0].check_literal(state, debug=debug) or self.child_clauses[1].check_literal(state, debug=debug)
        else: #this is atom
            if self.is_compare:
                items = self.literal_str.split(" ")
                assert len(items) == 3, "Literal.check_literal func: equal predicate has wrong format: {}".format(self.literal_str)
                return items[1] == items[2]
            else:
              if self.literal_str not in state and debug:
                print("[line 1570]: False bc not having {} in state".format(self.literal_str))
              return self.literal_str in state

class HTN:
  '''
  this class contain information about the goal tasks, which are specified in the problem file after '(:htn'
  - initialized with the htn string (the string after '(:htn' in problem.hddl)
  - self.reset(): reset the htn instance into its initial state
  - self.parse_task(): analyze the str and parse tasks in to list of tasks and ordering
  - self.update_htn(env_dictionary, world_state): update the htn instance based on the world_state
  '''
  def __init__(self, htn_str):
    self.htn_str = htn_str
    self.htn_headers = ['tasks', 'ordering','constraints', 'subtasks', 'ordered-subtasks']
    self.com_dict = split_components(self.htn_str, self.htn_headers, split_character=':')
    self.tasks = []
    self.tasks_ids = []
    self.ordering = []
    self.parse_task()
    self.constraints = [] #ignore for now
    self.completed_tasks = []
    self.remaining_tasks = copy.deepcopy(self.tasks)
    self.pending_tasks = []

  def __str__(self):
    return self.htn_str

  def reset(self):
    '''reset goal to initial state
    '''
    self.pending_tasks = []
    self.remaining_tasks = copy.copy(self.tasks)
    self.completed_tasks = []

  def parse_task(self):
    ''' parsing tasks
    considering cases: (1) ordered-subtasks, (2) subtasks + ordering, (3) subtasks (no ordering), (4) tasks + ordering, (5) tasks (without ordering)
    => read and return: list of subtasks, and list of ordering logics in pair e.g. [(t1,t2), (t2,t3), (t3,t4)]
    '''
    if len(self.com_dict['ordered-subtasks']) > 0:
      # ordered-subtasks appear in format of conjuntive list of subtasks or a single subtask, no need subtask-id
      for i in range(len(self.com_dict['ordered-subtasks'])):
        ordered_subtasks_str = self.com_dict['ordered-subtasks'][i]
        ordered_subtasks_str = clean_str(ordered_subtasks_str.split('ordered-subtasks (')[1])
        #use Literal to analyze the string:
        subtask_literal = Literal(ordered_subtasks_str)
        if subtask_literal.is_atom:
          self.tasks.append(subtask_literal.literal_str)
        elif subtask_literal.is_and:
          for j, child in enumerate(subtask_literal.child_clauses):
            assert child.is_atom, 'check ordered-subtasks format of method {}!!!'.format(self.name)
            self.tasks.append(child.literal_str)
            if j>0:
              self.ordering.append((subtask_literal.child_clauses[j-1].literal_str, child.literal_str))
        else:
          print("ERROR: wrong format for ordered-subtasks", ordered_subtasks_str)

    elif len(self.com_dict['subtasks']) > 0:
      for ii in range(len(self.com_dict['subtasks'])): # some prob has multiple ':subtasks' in htn
        subtasks_str = clean_str(self.com_dict['subtasks'][ii].split('subtasks (')[1])
        subtask_literal = Literal(subtasks_str)
        if subtask_literal.is_atom:
          self.tasks.append(subtask_literal.literal_str)
        elif subtask_literal.is_and:
          for child in subtask_literal.child_clauses:
            assert child.is_atom, 'Check subtask format of method {}'.format(self.name)
            if '(' in child.literal_str:
              subtask_id = child.literal_str.split(' ')[0]
              subtask = child.literal_str.split('(')[1]
              subtask = subtask.split(')')[0]
            else:
              subtask_id = None
              subtask = child.literal_str
            self.tasks_ids.append(subtask_id)
            self.tasks.append(subtask)
        else:
          print("ERROR: wrong format for subtasks in method:", self.name)

    elif len(self.com_dict['tasks']) > 0:
      subtasks_str = clean_str(self.com_dict['tasks'][0].split('tasks (')[1])
      subtask_literal = Literal(subtasks_str)
      if subtask_literal.is_atom:
        self.tasks.append(subtask_literal.literal_str)
      elif subtask_literal.is_and:
        for child in subtask_literal.child_clauses:
          assert child.is_atom, 'Check subtask format of method {}'.format(self.name)
          if '(' in child.literal_str:
            subtask_id = child.literal_str.split(' ')[0]
            subtask = child.literal_str.split('(')[1]
            subtask = subtask.split(')')[0]
          else:
            subtask_id = None
            subtask = child.literal_str
          self.tasks_ids.append(subtask_id)
          self.tasks.append(subtask)
      else:
        print("ERROR: wrong format for subtasks in method:", self.name)

    #handling ordering:
    if len(self.com_dict['ordering']) > 0:
      if len(clean_str(self.com_dict['ordering'][0])) > 0:
        ordering_str = clean_str(self.com_dict['ordering'][0]).split('ordering (')[1]
        if len(ordering_str.split(')')[0]) == 0:
          return
        ordering_literal = Literal(ordering_str)
        # Note that ordering_str should be in format of 'and (< task_id1 task_id2) (...)...'
        if ordering_literal.is_atom:
          self.ordering.append((ordering_literal.literal_str.split(' ')[1], ordering_literal.literal_str.split(' ')[2]))

  def update_htn(self,env_dictionary, world_state):
    '''This function update the htn instance by checking the world_state
    input:
    - env_dictionary: dictionary of environment
    - world_state: list of string of predicates
    output: (void)
    - update the completed_tasks, pending_tasks, and remaining_tasks
    '''

    completed_grounded_tasks = check_subtask_with_world_state(env_dictionary, world_state, self.tasks)
    for task in completed_grounded_tasks:
      if task not in self.completed_tasks:
        self.completed_tasks.append(task)
      if task in self.pending_tasks:
        self.pending_tasks.remove(task)
      if task in self.remaining_tasks:
        self.remaining_tasks.remove(task)
###### END OF CLASSES #######

