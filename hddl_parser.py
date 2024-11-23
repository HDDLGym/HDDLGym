from hddl_utils import read_hddl, split_components, generate_lifted_predicate
from hddl_utils import extract_full_str, extract_dynamic_predicates, generate_object_dict
from hddl_utils import HTN, Method, Action, clean_str, extract_object_list, Task
# Main parser function:

def main_parser(domain_file, prob_file, debug=False):
    '''
    inputs: domain and problem filename (strings of directories of the files)
    outputs: dictionary of following elements:
    - list of lifted predicates
    - list of types
    - list of objects (including agents)
    - list of agent's indices from objects_list (each agent perform 1 action at each step)
    - dictionary of type_object (including supertype: objects_list)
    - dictionary of supertype_type
    - list of grounded predicates
    - list of indices of usable grounded predicates
    - list of task instances
    - list of method instances
    - list of action instances
    '''
    domain_hddl = read_hddl(domain_file)
    prob_hddl = read_hddl(prob_file)
    domain_headers = ['requirements','types','constants','predicates','task','method','action']
    problem_headers = ['objects', 'htn', 'init', 'goal']
    domain_com_dict = split_components(domain_hddl, domain_headers, split_character = '(:')
    prob_com_dict = split_components(prob_hddl, problem_headers, split_character = '(:')

    env = dict()

    # 1. Generate list of lifted predicates (from domain_hddl):
    lifted_predicate_list = generate_lifted_predicate(domain_com_dict['predicates'][0])
    if debug: print('len(lifted_predicate_list) =',len(lifted_predicate_list))
    env['lifted predicate list']=lifted_predicate_list

    # 1.2 Generate list of lifted dynamic predicates (predicates that may change throughout effect)
    effect_segments = domain_hddl.split(":effect ")[1:]
    list_effect_str = []
    for effect_seg in effect_segments:
      list_effect_str.append(extract_full_str(effect_seg))
    # print("effect str:",list_effect_str)
    dynamic_pre_indices = extract_dynamic_predicates(lifted_predicate_list,list_effect_str)
    dynamic_pre_list = []
    for ind in dynamic_pre_indices:
      dynamic_pre_list.append(lifted_predicate_list[ind])
    env['lifted dynamic predicate list'] = dynamic_pre_list

    # 2. Generate list of types (domain_hddl)
    # 3. Generate list of objects (domain and problem)
    if len(domain_com_dict['constants'])>0:
      constants_str = domain_com_dict['constants'][0]
    else:
      constants_str = ''
    type_object_dict, supertype_type_dict = generate_object_dict(domain_com_dict['types'][0], prob_com_dict['objects'][0], constant_str = constants_str)
    types_list = list(type_object_dict.keys())
    objects_list = set()
    for t in types_list:
      objects_list.update(type_object_dict[t])
    objects_list = sorted(list(objects_list))
    env['types list'] = types_list
    env['objects list'] = objects_list
    env['type object dict'] = type_object_dict
    env['supertype type dict'] = supertype_type_dict
    # find all agent types:
    agent_types = set()
    if 'agent' in env['types list']:
      agent_types.add('agent')
    else:
      print("ERROR: no type 'agent', please specify 'agent' type in domain file!!!")
  
    if 'agent' in env['supertype type dict'].keys():
      agent_types.update(env['supertype type dict']['agent'])
    
    if debug:
      print("len of objects list:", len(env['objects list']))
      print('type_object_dict from main_parser:',type_object_dict)
      print('agent types: ', agent_types)

    # 4. Generate list of grounded predicates (domain, problem)
    grounded_predicate_list = []
    grounded_dynamic_predicate_list = []
    for id,lp in enumerate(lifted_predicate_list):
      gp = lp.generate_grounded_predicate(type_object_dict)
      grounded_predicate_list += gp
      if id in dynamic_pre_indices:
        grounded_dynamic_predicate_list += gp
    if debug:
      print("number of grounded predicates:", len(grounded_predicate_list))
      print('Number of grounded dynamic predicates:', len(grounded_dynamic_predicate_list))
      # print("grounded predicate list: \n",'\n'.join(grounded_predicate_list))

    env['grounded predicate list'] = grounded_predicate_list
    env['grounded dynamic predicate list'] = grounded_dynamic_predicate_list

    # 5. Initiate task instances, generate grounded task list
    lifted_tasks_list = []
    grounded_tasks_list = []
    for t_str in domain_com_dict['task']:
      lifted_tasks_list.append(Task(t_str))
      grounded_tasks_list += lifted_tasks_list[-1].generate_grounded_tasks(type_object_dict)
    if debug:
      print("number of grounded tasks:", len(grounded_tasks_list))

    env['lifted tasks list'] = lifted_tasks_list
    env['grounded tasks list'] = grounded_tasks_list

    # 6. Initiate method instances, generate grounded method list
    lifted_methods_list = []
    grounded_methods_list = []
    for m_str in domain_com_dict['method']:
      lifted_methods_list.append(Method(m_str, types_list, task_list=lifted_tasks_list))
      grounded_methods_list += lifted_methods_list[-1].generate_grounded_methods(type_object_dict)

    if debug:
      print("number of grounded methods:", len(grounded_methods_list))
      # print("grounded method list: \n",'\n'.join(grounded_methods_list))

    env['lifted methods list'] = lifted_methods_list
    env['grounded methods list'] = grounded_methods_list

    # 7. Initiate action instances, generate grounded action list
    lifted_actions_list = []
    grounded_actions_list = []
    for la in domain_com_dict['action']:
      lifted_actions_list.append(Action(la))
      grounded_actions_list += lifted_actions_list[-1].generate_grounded_actions(type_object_dict)

    env['lifted actions list']=lifted_actions_list
    env['grounded actions list'] = grounded_actions_list

    env['grounded operators list'] = env['grounded tasks list'] + env['grounded methods list'] + env['grounded actions list']
    env['lifted operators list'] = env['lifted tasks list'] + env['lifted methods list'] + env['lifted actions list']

    # Find a list of environment actions, which are not binding to any agents:
    lifted_env_action_list = []
    grounded_env_action_list = []
    for action in env['lifted actions list']:
      env_action_label = True
      for a_para_type in action.parameters_type_list:
        if a_para_type in agent_types:
          env_action_label = False
          break
      action.is_environment_action = env_action_label
      if env_action_label:
        lifted_env_action_list.append(action)
        grounded_env_action_list += action.generate_grounded_actions(type_object_dict)

    env['lifted environment actions list'] = lifted_env_action_list
    env['grounded environment actions list'] = grounded_env_action_list

    if debug:
      print("number of grounded actions:", len(grounded_actions_list))
      # print("grounded actions list: \n",'\n'.join(grounded_actions_list))
      print("number of grounded operators:", len(env['grounded operators list']))
      print("number of lifted operators: ",len(env['lifted operators list']))
      print("number of grounded environment actions list:",len(env['grounded environment actions list']))
      print("- lifted environment actions list:")
      for a in env['lifted environment actions list']:
        print(a.print_name())

    # 9. Find grounded static predicates of the problem (predicates that are never mentioned in effects, but appear in :init of problem)

    # 10. Create usable grounded predicates list by pruning the uncnecessary grounded predicate in 4 (remove un-mentioned static predicates)

    # 11. Generate list of indices of predicates in the initial state
    initial_state = []
    init_str = clean_str(prob_com_dict['init'][0].split('init')[1])
    pred_init = init_str.split('(')
    for pre in pred_init:
      if ')' in pre:
        initial_state.append(pre.split(')')[0])
    env['initial state'] = initial_state

    # 12. Generate list of goals (either task or predicates)
    htn_goal = HTN(prob_com_dict['htn'][0])
    env['htn goal'] = htn_goal

    return env


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
  for ind_lifted, lifted in enumerate(lifted_list):
    if lifted.name in grounded_str:
      lifted_indices.append(ind_lifted)

  for ind_obj, obj in enumerate(object_list):
    if obj in grounded_str:
      object_indices.append(ind_obj)

  return lifted_indices, object_indices

def possible_method_of_task_action(world, grounded_task_action):
  '''this function provide a list of all grounded methods that require the grounded_task_action
  inputs:
  - world: HDDLEnv, only use world.env dict and world.state
  - grounded_task_action: grounded task or action

  outputs:
  - related_grounded_methods: list of related grounded methods

  Approach:
  1. find all lifted method that has task/action name in their subtask list
  2. for each related method, generate the grounded list of methods according to the objects in task/action;
      and append that list to the related_grounded_methods
  '''
  related_grounded_methods = []
  # related_lifted_methods_with_obj_dict = []
  target_subtask_name, target_subtask_obj = extract_object_list(grounded_task_action)
  for m in world.env['lifted methods list']:
    for subtask in m.subtasks:
      sub_name, sub_var = extract_object_list(subtasks)
      if sub_name == target_subtask_name:
        custom_var_obj_dict = dict()
        for i in range(len(sub_var)):
          custom_var_obj_dict[sub_var[i]] = target_subtask_obj[i]
        # related_lifted_methods_with_obj_dict.append((m, custom_var_obj_dict))
        related_grounded_methods += m.generate_grounded_methods(world.env['type object dict'], task_var_object_dict = custom_var_obj_dict)

  return related_grounded_methods
