# This code is to experiment how to extract effects for eacht task
import re
from modify_hddl_domain_for_HDDLGym import remove_comments_from_hddl
import copy

domain_file = "./input_overcooked_task_effect_test.hddl"
output_file = "./output_overcooked_task_effect_test.hddl"

domain_file = "./ipc2023_domains/Rover/domain.hddl"


class Task:
        def __init__(self, task_block):
                self.block = task_block
                self.parse_task_block()
                self.effect_generated = False
                self.effect_block = ""
                self.new_task_block_with_effect = ""
                

        def parse_task_block(self,):
                """
                """
                name_match = re.search(r'\(:task\s+([^\s\(]+)', self.block)
                self.name = name_match.group(1)
                parameter_match = re.search(r':parameters\s*\(([^)]+)\)', self.block)
                self.parameters = parameter_match.group(1)
                self.parameter_name, self.parameter_type = self.parse_parameters()

        def parse_parameters(self):
            tokens = self.parameters.strip().split()
            param_names = []
            param_types = []

            current_group = []

            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token.startswith('?'):
                    current_group.append(token)
                    i += 1
                elif token == '-' and i + 1 < len(tokens):
                    type_ = tokens[i + 1]
                    for param in current_group:
                        param_names.append(param)
                        param_types.append(type_)
                    current_group = []
                    i += 2
                else:
                    # Skip unexpected token
                    i += 1

            return param_names, param_types
        
        def generate_effect(self, domain_file, task_list, task_name_list, debug=False):
            '''Generate effect for task, by going thru all methods
            # TODO: check parameter mapping between task, subtask, match_task/action.
            '''
            # if self.name == "navigate_abs": debug=True
            if debug:
                print("*** Generate task effects for task {}...".format(self.name))
                # print("task name list", task_name_list)
            self.effect_block = ""
            method_blocks = extract_blocks(filepath=domain_file, block_type="\(:method")
            action_blocks = extract_blocks(filepath=domain_file, block_type="\(:action")
            add_effects_set_list = list()
            del_effects_set_list = list()
            for method_block in method_blocks:
                if ":task ({}".format(self.name) not in method_block:
                    continue
                else:
                    # if "empty-store-1" in method_block:
                    #     debug = False
                    # Now, generate a dictionary to map method params to original task params
                    method_to_original_task_dict = generate_dictionary_mapping_method_params_to_task_action_params(self.block, method_block, suffix='_original_task')
                    add_effects_m = set()
                    del_effects_m = set()
                    # First, let precondition also be effects
                    precondition_block_match = re.search(r':precondition\s+(.*?)\s*(?=:[a-z\-]+|\)\s*$)', method_block, re.DOTALL)
                    if precondition_block_match:
                        precondition_block = precondition_block_match.group(1).strip()
                        if debug:
                            print("DEBUG: precondition of method {} is \n{}".format(method_block[10:25],precondition_block))
                        precondition_add_effects_m, precondition_del_effects_m = extract_add_and_del_effects_from_effect_block(precondition_block)

                    subtasks = extract_subtasks(method_block) # list of dictionaries
                    if len(subtasks) == 0 and precondition_block_match:
                        # if there is no subtasks, then precondition will be added into effect list:
                        add_effects_m.update(precondition_add_effects_m)
                        del_effects_m.update(precondition_del_effects_m)
                    # print("subtasks of method {} is {}".format(method_block[0:25], subtasks))
                    skip_this_method = False
                    for subtask in subtasks:
                        if subtask['name'] == self.name: #this task is included in the subtask list
                            # print("Skipping this method {}".format(method_block[:20]))
                            skip_this_method = True
                            break
                        if subtask['name'] in task_name_list: # this subtask is a task
                            match_task = task_list[task_name_list.index(subtask['name'])]
                            if not match_task.effect_generated:
                                match_task.generate_effect(domain_file, task_list, task_name_list)
                            # this subtask is a task that already (or should) has effects
                            # Now, generate a dictionary to map method params to subtask task_params
                            method_to_task_params_dict = generate_dictionary_mapping_method_params_to_task_action_params(match_task.block, method_block, suffix="_effect_task")
                            task_to_method_params_dict = reverse_mapping(method_to_task_params_dict)
                            # Extract effects, ensuring the params in effects are method_params
                            add_effs, del_effs = extract_add_and_del_effects_from_effect_block(match_task.effect_block, task_to_method_params_dict) 
                            add_effects_m.update(add_effs)
                            del_effects_m.update(del_effs)
                        else: # subtask is an action
                            # print("Subtask {} is an action!".format(subtask['name']))
                            subtask_name = subtask['name']
                            matched_action_block = None
                            action_block_pattern = rf'\(:action\s+{re.escape(subtask_name)}'
                            for action_b in action_blocks:
                                # if action_block_pattern in action_b:
                                if re.search(action_block_pattern, action_b):
                                    matched_action_block = action_b
                                    break
                            assert matched_action_block != None, "Could not find action block for action {}".format(subtask_name)
                            # print("matched action block:", matched_action_block)
                            # action_block_pattern = rf'\(:action\s+{re.escape(subtask_name)}'
                            # print("Action block pattern:", action_block_pattern)
                            # action_blocks = extract_blocks(domain_file,block_type=action_block_pattern)
                            # assert len(action_blocks) == 1, "Action should be defined once in the domain."
                            method_to_action_params_dict = generate_dictionary_mapping_method_params_to_task_action_params(matched_action_block,method_block, subtask_dict=subtask, debug=debug)
                            action_to_method_params_dict = reverse_mapping(method_to_action_params_dict)
                            add_effs, del_effs = extract_effects_from_action_block(matched_action_block, action_to_method_params_dict, debug=debug)
                            add_effects_m.update(add_effs) # note: all effects are with method_params
                            del_effects_m.update(del_effs)
                    if skip_this_method:
                        continue
                    else:
                        if debug:
                            print("DEBUG: before cleaning: add_effects_m for task {} from method {}: {}".format(self.name,method_block.split(":method ")[1].split()[0], add_effects_m))
                            print("DEBUG: before cleaning: del_effects_m for task {} from method {}: {}".format(self.name, method_block.split(":method ")[1].split()[0], del_effects_m))
                        cleaned_add_effects_m = add_effects_m - del_effects_m
                        cleaned_del_effects_m = del_effects_m - add_effects_m

                        if len(cleaned_add_effects_m) + len(cleaned_add_effects_m) == 0 and precondition_block_match:
                            # Add precondition_effects to an empty list of effects after going thru subtask:
                            if debug:
                                print("DEBUG: Add precondition as effect bc no effects left after doing subtasks")
                            cleaned_add_effects_m.update(precondition_add_effects_m)
                            cleaned_del_effects_m.update(precondition_add_effects_m)
                        # convert to task_param:
                        cleaned_translated_add_effects_m = []
                        cleaned_translated_del_effects_m = []
                        for eff in cleaned_add_effects_m:
                            cleaned_translated_add_effects_m.append(_map_effect_parameters(eff, method_to_original_task_dict))
                        for del_eff in cleaned_del_effects_m:
                            cleaned_translated_del_effects_m.append(_map_effect_parameters(del_eff, method_to_original_task_dict))

                        add_effects_set_list.append(set(cleaned_translated_add_effects_m))
                        del_effects_set_list.append(set(cleaned_translated_del_effects_m))

            if debug:
                print("Add_effect_set_list for task {}: {}".format(self.name, add_effects_set_list))
            # Now, find the common predicates of add_effects and del_effects
            common_add_effects = set.intersection(*add_effects_set_list)
            common_del_effects = set.intersection(*del_effects_set_list)
            # cleaning:
            self.add_effects = common_add_effects - common_del_effects
            self.del_effects = common_del_effects - common_add_effects

            # Validation:
            # Check if parameters in the effect block are mentioned in parameters
            task_parameter_with_original_task_label = set()
            # print("self.parameter_name:",self.parameter_name)
            for para in self.parameter_name:
                task_parameter_with_original_task_label.add(para+"_original_task")
            # print("task_parameter_with_original_task_label",task_parameter_with_original_task_label)
            self.add_effects = filter_effects_by_parameters(self.add_effects, task_parameter_with_original_task_label)
            self.del_effects = filter_effects_by_parameters(self.del_effects, task_parameter_with_original_task_label)

            if len(self.add_effects) + len(self.del_effects) == 0:
                print("No effect found for the task {}!\n To debug: you might want to add more parameters or more methods to achieve this task!".format(self.name))

            self.effect_block = generate_effect_block(self.add_effects, self.del_effects)
            # remove '_original_task'
            self.effect_block = self.effect_block.replace("_original_task", "")
            # print("*** Effect Block for task {} is \n{}\n".format(self.name, self.effect_block))
            self.effect_generated = True
            self.add_effect_to_task_block(domain_file)

        def add_effect_to_task_block(self, domain_file) -> str:
            # Remove any trailing whitespace and find last closing parenthesis
            task_block = self.block.strip()

            if ':effect' in task_block:
                raise ValueError("Task block already contains an :effect block")

            # Find indentation of the (:task line
            with open(domain_file, 'r') as f:
                for line in f:
                    if f'(:task {self.name}' in line:
                        base_indent = re.search(rf'^( *)\(:task\s*{self.name}', line).group(1)
                        break
            
            # task_line_match = re.search(r'^( *)\(:task', self.block, re.MULTILINE)
            # if task_line_match: print("task line match:---{}+++".format(task_line_match.group(1)))
            # base_indent = task_line_match.group(1) if task_line_match else ""

            # One level deeper indentation (e.g., +1 tab or +2 spaces)
            deeper_indent = base_indent + "  "  # 4-space indent level; adjust if needed
            effect_lines = self.effect_block.split('\n')
            effect_bl_indent = '\n'+ deeper_indent
            effect_bl = effect_bl_indent.join(effect_lines)

            if task_block.endswith(')'):
                insertion_point = task_block.rfind(')')
                # Insert the indented effect block before the final closing paren
                new_task_block = (
                    task_block[:insertion_point] + '\n'+
                    deeper_indent + effect_bl +
                    task_block[insertion_point:]
                )
            else:
                raise ValueError("Invalid task block format — no closing parenthesis found")

            self.new_task_block_with_effect = new_task_block
            
            
        
# OTHER FUNCTIONS:
def extract_blocks(filepath, block_type="\(:method"):
    blocks = []
    with open(filepath, 'r') as file:
        lines = file.readlines()

    inside_block = False
    paren_count = 0
    current_block = []

    for line in lines:
        stripped = line.strip()

        if not inside_block and re.search(rf"\s*{block_type}([\s\)])", stripped):# stripped.startswith(f"{block_type}"):
            inside_block = True
            paren_count = stripped.count('(') - stripped.count(')')
            current_block = [stripped]
            if paren_count == 0:
                blocks.append('\n'.join(current_block))
                inside_block = False
                current_block = []
        elif inside_block:
            paren_count += stripped.count('(') - stripped.count(')')
            current_block.append(stripped)
            if paren_count == 0:
                blocks.append('\n'.join(current_block))
                inside_block = False
                current_block = []

    return blocks

def extract_subtasks_old(method_block: str) -> list:
    # Match either :subtasks or :ordered-subtasks
    match = re.search(r':(?:ordered-)?subtasks\s+\(and\s+(.*?)\)(?=\s*:[a-z\-]+|\s*\)\s*$)', method_block, re.DOTALL)
    
    if not match:
        print("no subtasks found for method str:", method_block)
        return []  # No subtasks found

    subtasks_raw = match.group(1)
    print("subtasks_raw:", subtasks_raw)

    # Find all task lines of the form (task-name (task args))
    task_matches = re.findall(r'\(([^()\s]+)\s+\(([^()]+)\)\)', subtasks_raw)

    subtasks = []
    for label, task_expr in task_matches:
        task_parts = task_expr.split()
        task_name = task_parts[0]
        task_args = task_parts[1:]
        subtasks.append({
            "label": label,
            "task": task_name,
            "args": task_args
        })

    return subtasks


def extract_subtasks(method_block: str):
    subtasks = []

    # Match entire :subtasks or :ordered-subtasks block
    match = re.search(
        r':(?P<type>ordered-)?subtasks\s+\((.*?)\)(?=\s*:[a-z\-]+|\s*\)\s*$)',
        method_block,
        re.DOTALL
    )
    if not match:
        return subtasks

    is_ordered = bool(match.group('type'))
    subtask_block = match.group(2).strip()
    if len(subtask_block) == 0:
        return subtasks

    # Check if it's (and (...)) or just a single subtask
    if subtask_block.startswith("and"):
        # Remove "and" and parse individual subtasks
        subtask_block = subtask_block[3:].strip()

        if is_ordered:
            # Case: ordered-subtasks with (and ...)
            matches = re.findall(r'\(([^()\s]+(?:\s+[^\(\)\s]+)+)\)', subtask_block)
            for task_str in matches:
                parts = task_str.strip().split()
                subtasks.append({
                    'name': parts[0],
                    'string': ' '.join(parts)
                })
        else:
            # Case: subtasks with IDs and (and ...)
            matches = re.findall(r'\(([^()\s]+)\s+\(([^()]+)\)\)', subtask_block)
            for label, task_expr in matches:
                parts = task_expr.strip().split()
                subtasks.append({
                    'name': parts[0],
                    'string': ' '.join(parts)
                })
    else:
        # Case: single subtask (no 'and')
        match = re.match(r'\(([^()\s]+)\s+\(([^()]+)\)\)', f"({subtask_block})")
        if not match:
            # Single task without ID
            if not len(subtask_block) ==0:
                parts = subtask_block.strip().split()
                subtasks.append({
                    'name': parts[0],
                    'string': ' '.join(parts)
                })
        else: # single task with ID
            label, task_expr = match.groups()
            parts = task_expr.strip().split()
            subtasks.append({
                'name': parts[0],
                'string': ' '.join(parts)
            })

    return subtasks


def extract_effects_from_action_block(action_block: str, param_mapping:dict=None, debug=False):
    '''Get add effects and del effects from action block, convert parameter if provide param_mapping (to match with method params)
    inputs:
    - action_block: string defining action, (:action action_name ...)
    - param_mapping: a dictionary {action_param: method_param}
    outputs:
    - add_effects: set of positive effects
    - del_effects: set of negative effects
    '''
    add_effects = set()
    del_effects = set()
    
    # Extract the full effect block
    effect_match = re.search(r':effect\s+(.*?)\s*(?=:[a-z\-]+|\)\s*$)', action_block, re.DOTALL)
    if not effect_match:
        return add_effects, del_effects

    effect_block = effect_match.group(1).strip()
    add_effects, del_effects = extract_add_and_del_effects_from_effect_block(effect_block, param_mapping=param_mapping)
    if debug:
        print("DEBUG: List of add and del effects from action {} is:".format(action_block.split(':action ')[1].split()[0]),(add_effects, del_effects))
    return add_effects, del_effects

def extract_add_and_del_effects_from_effect_block(effect_block:str, param_mapping: dict = None):
    add_effects = set()
    del_effects = set()
    # print("---effect_block:",effect_block)
    effect_block = effect_block.strip()
    # Remove outer (and ...) if present
    if effect_block.startswith('(and'):
        effect_block = effect_block[4:].strip()
        if effect_block.endswith(')'):
            effect_block = effect_block[:-1].strip()
    elif effect_block.startswith(':effect (and'):
        start_point = len(":effect (and")
        effect_block = effect_block[start_point:].strip()
        if effect_block.endswith(')'):
            effect_block = effect_block[:-1].strip()
    # print("---after effect_block:",effect_block)

    # Match all (not ...) and other predicates
    tokens = re.findall(r'\(not\s+(\([^)]+\))\)|(\([^)]+\))', effect_block)

    for neg, pos in tokens:
        if neg:
            effect_str = neg.strip()
            if param_mapping:
                effect_str = _map_effect_parameters(effect_str, param_mapping)
            del_effects.add(effect_str)
        elif pos:
            effect_str = pos.strip()
            if param_mapping:
                effect_str = _map_effect_parameters(effect_str, param_mapping)
            add_effects.add(effect_str)

    return add_effects, del_effects

def _map_effect_parameters(effect_str: str, mapping: dict) -> str:
    """
    Replaces each parameter in the effect string according to the mapping.
    Example: (at ?p ?l) with {?p: ?pkg, ?l: ?loc} → (at ?pkg ?loc)
    """
    return re.sub(r'\?[\w\d_-]+', lambda m: mapping.get(m.group(), m.group()), effect_str)

def generate_effect_block(add_effects: set, del_effects: set) -> str:
    # Remove conflicting literals (in both add and delete sets)
    clean_adds = add_effects - del_effects
    clean_dels = del_effects - add_effects

    effect_parts = []

    for eff in sorted(clean_adds):
        effect_parts.append(eff)

    for eff in sorted(clean_dels):
        effect_parts.append(f"(not {eff})")

    if len(effect_parts) == 0:
        return ":effect ()"
    elif len(effect_parts) == 1:
        return f":effect {effect_parts[0]}"
    else:
        return ":effect (and\n  " + "\n  ".join(effect_parts) + "\n)"

def filter_effects_by_parameters(effects: set, valid_params: set) -> set:
    filtered = set()
    for eff in effects:
        # Extract variables like ?p, ?v
        used_params = set(re.findall(r'\?[\w\d_-]+', eff))
        if used_params.issubset(valid_params):
            filtered.add(eff)
        # else:
        #     print("This effect use different parameter from task params: ",eff)
    return filtered



def generate_dictionary_mapping_method_params_to_task_action_params(task_action_block: str, method_block: str, suffix='',subtask_dict=None, debug=False) -> dict:
    # 1. Extract task/action name and parameters from task_block
    header_match = re.search(r'\(:(?:task|action)\s+([^\s]+)\s+:parameters\s+\((.*?)\)', task_action_block, re.DOTALL)
    if not header_match:
        raise ValueError("Invalid task_block format")

    task_name = header_match.group(1)
    task_param_raw = header_match.group(2).split()
    
    # Extract just parameter names (skip types and dashes)
    task_params = []
    i = 0
    while i < len(task_param_raw):
        if task_param_raw[i].startswith('?'):
            task_params.append(task_param_raw[i])
            i += 1
        elif task_param_raw[i] == '-':
            i += 2  # skip type
        else:
            i += 1

    # 2. Find the (task_name ...) line in method_block or in subtask_dict if provided (in case task/action are repeated in the method)
    if subtask_dict != None:
        method_params_used_in_task = subtask_dict['string'].split()[1:]
        if debug:
            print('DEBUG: method_params_used_in_task/action: ', method_params_used_in_task)
    else:
        task_call_match = re.search(rf'\(\s*{re.escape(task_name)}\s+([^)]+)\)', method_block)
        if not task_call_match:
            raise ValueError(f"No task call found for '{task_name}' in method_block")

        method_params_used_in_task = task_call_match.group(1).split()

    # 3. Extract all method parameters declared
    method_param_match = re.search(r':parameters\s*\(([^)]+)\)', method_block)
    method_param_raw = method_param_match.group(1).split() if method_param_match else []
    method_params = []
    i = 0
    while i < len(method_param_raw):
        if method_param_raw[i].startswith('?'):
            method_params.append(method_param_raw[i])
            i += 1
        elif method_param_raw[i] == '-':
            i += 2
        else:
            i += 1

    # 4. Build mapping dictionary
    mapping = {}
    for i, method_param in enumerate(method_params_used_in_task):
        if i < len(task_params):
            mapping[method_param] = task_params[i] + suffix

    for method_param in method_params:
        if method_param not in mapping:
            mapping[method_param] = method_param  # map to itself

    return mapping

def reverse_mapping(mapping: dict) -> dict:
    return {v: k for k, v in mapping.items()}

def generate_task_effect(domain_file):
    '''
    '''
    #Clean up comments:
    domain_file_original = copy.copy(domain_file)
    domain_file = domain_file.replace(".hddl","_nocomments.hddl")
    remove_comments_from_hddl(domain_file_original, domain_file)
    # method_blocks = extract_blocks(filepath=domain_file, block_type="(:method")
    task_blocks = extract_blocks(domain_file, block_type="\(:task")
    # print("task_blocks:",task_blocks)
    task_list = []
    task_name_list = []
    for task_block in task_blocks:
        t = Task(task_block)
        task_list.append(t)
        task_name_list.append(t.name)
    for task in task_list:
        if not task.effect_generated:
            task.generate_effect(domain_file, task_list, task_name_list)

    # Update domain with task with effects:
    with open(domain_file, 'r') as old_f:
        domain_str = old_f.read()
    
    for task in task_list:
        domain_str = domain_str.replace(task.block, task.new_task_block_with_effect)
    # save new domain:
    new_domain_file = domain_file.replace(".hddl","_with_task_effect.hddl")
    with open(new_domain_file,'w') as new_f:
        new_f.write(domain_str)


if __name__ == "__main__":

    generate_task_effect(domain_file)


