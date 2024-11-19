import re
import os
from pathlib import Path
import argparse

def remove_comments_from_pddl(input_file, output_file):
    """
    Removes comments from a PDDL file and writes the result to a new file.
    
    Parameters:
    - input_file: Path to the input PDDL file.
    - output_file: Path to the output PDDL file without comments.
    """
    # Open the input PDDL file in read mode
    with open(input_file, 'r') as infile:
        lines = infile.readlines()
    
    # Process lines to remove comments (everything after ';')
    cleaned_lines = []
    for line in lines:
        # Match the indentation using a regex, then remove comments
        line = line.replace(' )',')')
        line = line.replace('( )','()')
        match = re.match(r'(\s*)(.*)', line)
        if match:
            indent = match.group(1)
            content = match.group(2)
            
            # Remove comments from the content
            cleaned_content = re.sub(r';.*', '', content).rstrip()
            
            # If there's still content left, preserve the indentation
            if cleaned_content:
                cleaned_lines.append(indent + cleaned_content)
    
    # Write the cleaned lines to the output file
    with open(output_file, 'w') as outfile:
        for line in cleaned_lines:
            outfile.write(line + '\n')

def remove_extra_whitespaces(input_file, output_file):
    with open(input_file, 'r') as file:
        content = file.readlines()

    cleaned_content = []
    
    for line in content:
        line = line.replace(' )',')')
        line = line.replace('( )','()')
        # Split the line by tabs to preserve indentation
        parts = line.split('\t')
        # Remove extra spaces from each part, then join by tab
        cleaned_line = '\t'.join(' '.join(part.split()) for part in parts)
        if cleaned_line.strip():  # Keep non-empty lines
            cleaned_content.append(cleaned_line)

    with open(output_file, 'w') as file:
        file.write('\n'.join(cleaned_content))

def extract_blocks(content, block_type):
    blocks = []
    stack = []
    start_idx = None

    # Iterate through the content character by character
    for i, char in enumerate(content):
        # print(content[i:i + len(block_type)])
        if content[i:i + len(block_type)] == block_type:# and (i == 0 or content[i-1].isspace()):
            start_idx = i  # Potential start of a block
            stack=[]
        
        if char == '(':
            stack.append('(')
        elif char == ')':
            if stack:
                stack.pop()  # Match a closing parenthesis

            if not stack and start_idx is not None:
                # We've matched all parentheses for this block
                blocks.append(content[start_idx:i + 1])
                start_idx = None

    return blocks



def check_hddl_format(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    errors = []
    
    
    # Check for balanced parentheses
    if content.count('(') != content.count(')'):
        errors.append("Unbalanced parentheses in the file.")
    
    # Required sections and keywords in PDDL
    required_keywords = ['domain', ':requirements', ':predicates', ':task', ':method', ':action', ':parameters', ':precondition', ':effect']
    
    for keyword in required_keywords:
        if keyword not in content:
            errors.append(f"Missing keyword: {keyword}")

    # Check parentheses structure
    stack = []
    for i, char in enumerate(content):
        if char == '(':
            stack.append(i)
        elif char == ')':
            if not stack:
                errors.append(f"Unmatched closing parenthesis at position {i}")
            else:
                stack.pop()
    
    if stack:
        errors.append(f"Unmatched opening parenthesis at positions: {stack}")

    action_pattern = r'\(:action\s+.*?\)'
    method_pattern = r'\(:method\s+.*?\)'
    task_pattern = r'\(:task\s+.*?\)'
    # Basic validation for action definitions
    # action_blocks = re.findall(action_pattern, content, re.DOTALL)
    action_blocks = extract_blocks(content, '(:action')
    for action in action_blocks:
        if ':parameters' not in action or ':precondition' not in action or ':effect' not in action:
            errors.append(f"Malformed action block: {action.strip()[:50]}...")
    
    # Basic validation for task definitions
    # task_blocks = re.findall(task_pattern, content, re.DOTALL)
    task_blocks = extract_blocks(content, '(:task')
    for task in task_blocks:
        if ':parameters' not in task or ':effect' not in task:
            errors.append(f"Malformed task block: {task.strip()[:]}...")

    # Basic validation for method definitions:
    # method_blocks = re.findall(method_pattern, content, re.DOTALL)
    method_blocks = extract_blocks(content, '(:method')
    for method in method_blocks:
        if ':parameters' not in method or ':task' not in method or (':subtasks' not in method and ':ordered-subtasks' not in method):
            errors.append(f"Malformed method block: {method.strip()[:50]}...")
            if ':parameters' not in method:
                print("Need parameters")
            if ':task' not in method:
                print("Need associated task")
            if ':subtasks' not in method or ':ordered-subtasks' not in method:
                print("Need subtasks")
    
    
    if not errors:
        print("HDDL file format is correct.")
    else:
        print("Errors found in PDDL file:")
        for error in errors:
            print(f"- {error}")

import re
from collections import defaultdict

def parse_types(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract the :types section using regex
    types_section = re.search(r':types\s+(.*?)(?=\(:|$)', content, re.DOTALL)
    
    if not types_section:
        print("No ':types' section found in the PDDL file.")
        return None

    # Flatten and clean types
    types_hierarchy = types_section.group(1).replace('\n', ' ').strip()
    
    # Build the type hierarchy
    type_hierarchy = defaultdict(list)
    for type_group in re.findall(r'(\S+(?:\s+-\s+\S+|))', types_hierarchy):
        types = type_group.split(' - ')
        subtype = types[0].strip()
        if len(types) > 1:
            supertype = types[1].strip()
            type_hierarchy[supertype].append(subtype)
        else:
            type_hierarchy['root'].append(subtype)  # root types with no supertype

    return type_hierarchy

def is_agent_in_hierarchy(type_hierarchy, target='agent'):
    # Check if 'agent' exists as a type or a supertype
    if target in type_hierarchy or any(target in subtypes for subtypes in type_hierarchy.values()):
        print(f"'{target}' is a type or supertype.")
    else:
        print(f"'{target}' is NOT a type or supertype.")

import re

def find_common_types(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Extract types from the :types section
    types_section = re.search(r':types\s+(.*?)(?=\(:|$)', content, re.DOTALL)
    all_types = set()
    if types_section:
        types = types_section.group(1).replace('\n', ' ').strip().split()
        all_types.update([t for t in types if t != '-'])

    # Extract all action blocks
    actions = extract_blocks(content, '(:action')
    type_mentions_per_action = []

    for action in actions:
        # Find all parameter types in the action block
        parameters = re.findall(r'\?[\w-]+\s+-\s+([\w-]+)', action)
        # Collect mentioned types in preconditions and effects
        preconditions_effects = re.findall(r'\(([\w-]+)\s', action)
        all_mentioned_types = set(parameters + preconditions_effects)
        type_mentions_per_action.append(all_mentioned_types)

    # Find types common to all actions
    if type_mentions_per_action:
        common_types = set.intersection(*type_mentions_per_action)
        relevant_common_types = common_types.intersection(all_types)
        return relevant_common_types
    else:
        print("No actions found in the PDDL file.")
        return set()

def find_empty_precondition_and_effect(file_path):
    none_actions = []
    with open(file_path, 'r') as file:
        content = file.read()
    action_blocks = extract_blocks(content, '(:action')
    # Extract all action blocks
    for action in action_blocks:
        none_actions += re.findall(r'\(:action\s+([\w-]+).*?:precondition\s*\(\s*\).*?:effect\s*\(\s*\)', action, re.DOTALL)

    if none_actions:
        print("Actions with empty preconditions and effects:")
        for action in none_actions:
            print(f" - {action}")
    else:
        print("No actions found with both empty preconditions and effects.")
    
    return none_actions

def add_agent_parameter(file_path, output_path):
    with open(file_path, 'r') as file:
        content = file.read()

    # Pattern to find :parameters section within each :action block
    action_pattern = r'(:action\s+[\w-]+\s*:parameters\s*\()([^\)]*)(\))'
    
    # Add the new parameter to the parameters section
    updated_content = re.sub(
        action_pattern,
        lambda match: f"{match.group(1)}{match.group(2).strip()} ?agent - agent{match.group(3)}",
        content
    )

    # Save the updated content to a new file
    with open(output_path, 'w') as file:
        file.write(updated_content)

    print(f"Updated PDDL file saved to {output_path}")

def get_all_subtypes(type_hierarchy, target_type):
    """Recursively get all subtypes of a given type."""
    subtypes = set(type_hierarchy.get(target_type, []))
    for subtype in list(subtypes):
        subtypes.update(get_all_subtypes(type_hierarchy, subtype))
    return subtypes

def find_actions_without_agent(file_path, agent_type='agent'):
    """Find actions that do not use 'agent' or its subtypes."""
    type_hierarchy = parse_types(file_path)
    agent_related_types = {agent_type}.union(get_all_subtypes(type_hierarchy, agent_type))

    with open(file_path, 'r') as file:
        content = file.read()

    # Extract all :action blocks
    action_pattern = re.compile(r'\(:action\s+([\w-]+).*?:parameters\s*\((.*?)\)', re.DOTALL)
    actions_without_agent = []

    for match in action_pattern.finditer(content):
        action_name = match.group(1)
        parameters = match.group(2)

        # Extract parameter types
        parameter_types = re.findall(r'\?[\w-]+\s+-\s+([\w-]+)', parameters)
        if not any(ptype in agent_related_types for ptype in parameter_types):
            actions_without_agent.append(action_name)

    return actions_without_agent

def add_agent_parameter_to_specific_actions(input_file_path, output_file_path, action_names):
    """
    Add ?agent - agent to the :parameters section of specific actions in a PDDL file.

    Parameters:
    - file_path: Path to the PDDL file.
    - action_names: List of action names to modify.
    """
    with open(input_file_path, 'r') as file:
        content = file.read()

    # Regex to match specific :action blocks and their :parameters sections
    action_pattern = re.compile(
        r'(:action\s+({})\s+.*?:parameters\s*\()(.*?)(\)\s*(:precondition|:effect))'.format('|'.join(map(re.escape, action_names))),
        re.DOTALL
    )

    # Define the new parameter to add
    new_parameter = '?agent - agent'

    def add_parameter(match):
        """Modify the :parameters section to include ?agent - agent."""
        action_name = match.group(2)  # Action name
        parameters = match.group(3)
        rest_of_action = match.group(4)

        # Check if the parameter is already present
        if new_parameter not in parameters:
            parameters = f"{parameters} {new_parameter}"

        return f"{match.group(1)}{parameters}{rest_of_action}"

    # Apply the modification to the specified actions
    modified_content = action_pattern.sub(add_parameter, content)

    # Write the modified content back to the file
    with open(output_file_path, 'w') as file:
        file.write(modified_content)

    print(f"Parameter '{new_parameter}' added to specified actions: {', '.join(action_names)}")


def rename_action_in_hddl(file_path, old_name, new_name):
    """
    Rename an action in an HDDL domain file and update corresponding method calls.

    Parameters:
    - file_path: Path to the HDDL domain file.
    - old_name: The current name of the action to be renamed.
    - new_name: The new name for the action.
    """
    with open(file_path, 'r') as file:
        content = file.read()

    # Rename the action itself
    content, action_renamed = re.subn(
        fr'\(:action\s+{re.escape(old_name)}\b',
        f'(:action {new_name}',
        content
    )

    # Rename all occurrences of the action in methods
    content, method_calls_updated = re.subn(
        fr'\({re.escape(old_name)}\b',
        f'({new_name}',
        content
    )

    # Check if changes were made
    if action_renamed == 0:
        print(f"Action '{old_name}' not found in the domain file.")
    else:
        print(f"Renamed action '{old_name}' to '{new_name}'.")

    if method_calls_updated > 0:
        print(f"Updated {method_calls_updated} method call(s) from '{old_name}' to '{new_name}'.")
    else:
        print(f"No method calls for action '{old_name}' were found.")

    # Write the updated content back to the file
    with open(file_path, 'w') as file:
        file.write(content)

    print("Changes saved to the domain file.")

def add_none_action_if_missing(input_file_path, output_file_path):
    """
    Adds a 'none' action to the PDDL domain file only if it doesn't already exist.
    
    The action will have:
    - Parameters: (?agent - agent)
    - Precondition: ()
    - Effect: ()
    """
    with open(input_file_path, 'r') as file:
        content = file.read()

    # Check if the 'none' action already exists
    if re.search(r'\(:action\s+none\b', content):
        print("The 'none' action already exists in the domain file.")
        return

    # Define the new 'none' action
    none_action = """
(:action none
  :parameters (?agent - agent)
  :precondition ()
  :effect ()
)
"""

    # Insert the new action before the last closing parenthesis of the file
    if content.strip().endswith(')'):
        updated_content = re.sub(r'\)\s*$', none_action + '\n)', content, count=1)
    else:
        updated_content = content + '\n' + none_action

    # Write the updated content back to the file
    with open(output_file_path, 'w') as file:
        file.write(updated_content)

    print("Added 'none' action to the domain file.")


def main_modify(input_file, output_file):
    '''
    Modify HDDL domain files to use HDDLGym
    1. remove comments
    2. Ensure agent is included in all agent-action. 
        If exist any non-agent action, ask users if they would like to add agent to the parameters of those actions
    3. Ensure at least 1 none action exist in the domain:
    '''

    # 1. remove comments
    remove_comments_from_pddl(input_file, output_file)

    # 2. Ensure agents is included in all agentic actions, otw, ask if users want to add agent to the parameters.
    no_agent_actions = find_actions_without_agent(output_file)
    print("no agent actions: ", no_agent_actions)
    add_agent_actions = []
    for no_agent_action in no_agent_actions:
        answered = False
        while not answered:
            add_agent = input("The action '{}' has no agent in its parameters, do you want to add agent \
                \n(if not, it will be an environmental action, aka no-agent action)? \n --- answer 'yes' or 'no'\n".format(no_agent_action))
            if add_agent.lower() == 'yes':
                add_agent_actions.append(no_agent_action)
                answered = True
            elif add_agent.lower()== 'no':
                answered = True
            else:
                print("*** Your answer, '{}', is INVALID! Answer should be 'yes' or 'no'! Please try again... ***".format(add_agent.lower()))
    
    add_agent_parameter_to_specific_actions(output_file, output_file, add_agent_actions)
    # 3. Ensure at least 1 none action exist in the domain:
    # Check if exist none action with empty precondition and effect, make sure the action name is 'none'
    none_actions = find_empty_precondition_and_effect(output_file)
    for none_action_name in none_actions:
        if none_action_name != 'none':
            rename_action_in_hddl(output_file, none_action_name, 'none')
    
    # Add none action if missing:
    add_none_action_if_missing(output_file, output_file)
    

def parse_arguments():
    '''Hyperparameter Setting'''
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser()
    parser.add_argument('--domain', type=str, default=str(script_dir / "test_input_modify_overcooked.hddl"), help='Which domain HDDL file to modify?')
    parser.add_argument('--new-domain', type=str, default=str(script_dir / "test_output_modify_overcooked.hddl"), help='What is new directory to save the modified domain?')
    opt = parser.parse_args()
    return opt

if __name__=="__main__":
    #unit test for the main_modify function:
    # folder = "./"
    # input_file_path = folder + 'test_input_modify_overcooked.hddl'  # Path to the input PDDL file
    # output_file_path = folder + 'test_output_modify_overcooked.hddl'  # Path for the cleaned output file
    opt = parse_arguments()
    input_file_path = opt.domain
    output_file_path = opt.new_domain
    main_modify(input_file_path, output_file_path)
    # Check if the domain is ready to use HDDLGym:
    check_hddl_format(output_file_path)
    
    # type_hierarchy = parse_types(output_file_path)
    # if type_hierarchy:
    #     is_agent_in_hierarchy(type_hierarchy)
    # common_types = find_common_types(output_file_path)
    # print(common_types)
    
    

        
