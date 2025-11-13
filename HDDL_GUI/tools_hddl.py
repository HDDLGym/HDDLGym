### HELPER FUNCTIONS FOR HDDL
import subprocess
import time
def removeMethod(domain_str, delete_method):
    """
    Remove a method block from the domain string by tracking parentheses.

    Args:
        domain_str (str): The domain string containing methods and tasks.
        delete_method (str): The name of the method to remove.

    Returns:
        str: The updated domain string with the specified method removed.
    """
    lines = domain_str.splitlines()
    updated_lines = []
    inside_method = False
    parentheses_count = 0

    for line in lines:
        # Check if the line starts the method to delete
        if not inside_method and f":method {delete_method}" in line:
            inside_method = True
            parentheses_count = line.count("(") - line.count(")")
            continue  # Skip this line (start of the method)

        # If inside the method block, track parentheses
        if inside_method:
            parentheses_count += line.count("(") - line.count(")")
            if parentheses_count <= 0:
                inside_method = False  # End of the method block
            continue  # Skip lines inside the method block

        # Add lines that are not part of the method block
        updated_lines.append(line)

    return "\n".join(updated_lines)

def extract_blocks(text, keyword="(:method"):
    blocks = []
    start = 0
    while True:
        start = text.find(f"{keyword}", start)
        if start == -1:
            break
        depth = 0
        for i in range(start, len(text)):
            if text[i] == '(':
                depth += 1
            elif text[i] == ')':
                depth -= 1
            if depth == 0:
                blocks.append(text[start:i+1])
                start = i + 1
                break
    return blocks
def updateDomain(domain_str, new_method=None, delete_method=None):
    ''' This function add new method into the domain file, and saved it in a different file with label -modifed-
    inputs:
    - domain_str: a string of domain
    - new_method: a string of (:method name...)
    outputs:
    new_domain_str: a string of updated domain with new_method
    '''
    updatedDomain = None 

    if new_method is not None and "(:method" in new_method:
        # extract method blocks: blocks that start with (:method:
        new_method_blocks_list = extract_blocks(new_method,keyword="(:method")
        # Find position to insert 
        ## find action block, add new_method right before the first action block
        i_metric = domain_str.find("(:action")
        if i_metric!=-1:
            i_insert = i_metric
        ## Else, insert before last parenthesis
        else:
            i_insert=len(domain_str)-1
            while domain_str[i_insert]!=')':
                i_insert-=1
        
        # Insert constraints into problem
        updatedDomain = domain_str[:i_insert] + "\n"+ "\n".join(new_method_blocks_list) + "\n" + domain_str[i_insert:]
    elif delete_method is not None:
        # remove method blocks: blocks that start with (:method:
        updatedDomain = removeMethod(domain_str, delete_method)
    else: 
        print("Error: No appropriate new method or delete method provided.")
        return domain_str
    
    return updatedDomain

def verifyMethodEncoding(updated_domain, problem, new_methods_str,current_path="./", debug = False, return_format_version=True):
    '''
    inputs:
    - updated_domain: string of updated domain with new methods
    - problem: string of problem in hddl
    - new_methods_str: string of new methods 
    - current_path: current path to save files
    outputs:
    - boolean if the updated domain is valid
    - feedback_str: string of feedback if it is not valid, return plan if it is valid
    '''
    # save updated_domain to a hddl file
    updated_domain_file_path = current_path + "new_domain.hddl"
    problem_file_path = current_path + "temp_problem.hddl"
    with open(updated_domain_file_path, "w") as updated_domain_file:
        updated_domain_file.write(updated_domain)
    with open(problem_file_path, "w") as problem_file:
        problem_file.write(problem)
    
    # run lilotane:
    try:
        start_time = time.time()
        output = subprocess.run(["./lilotane",updated_domain_file_path, problem_file_path], capture_output=True, text=True, check=True)
        end_time = time.time()
        planning_time = end_time-start_time
        output_str = output.stdout
        start_marker = "==>"
        end_marker = "<=="

        start = output_str.find(start_marker)
        end = output_str.find(end_marker, start)

        if start != -1 and end != -1:
            # Adjust to slice the content between markers
            extracted = output_str[start + len(start_marker):end].strip()
            raw_plan = extracted
            if return_format_version:
                extracted = format_lilotane_plan(extracted)
            if debug:
                print("Plan:\n", extracted)
                print(f"\n Planning time: {end_time-start_time:.6f}(sec)")
            extracted = extracted + f"\n--- Planning time: {planning_time:.6f}"
            return True, extracted
        else:
            print("Plan not found!")
            return False, "Couldn't find a valid plan, although there is no error."
    except subprocess.CalledProcessError as e:
        if new_methods_str == "":
            error = "Program failed! \n--- Exit code: "+ str(e.returncode) + "\nError output:"+ str(e.stderr)
        else:
            error = "Program failed with new method! New method added: \n{}".format(new_methods_str)+"\n--- Exit code: "+ str(e.returncode) + "\nError output:"+ str(e.stderr)
        return False, error
    
def format_lilotane_plan(raw_plan):
    '''
    '''
    lines = raw_plan.strip().split('\n')
    primitive_actions, task_methods = parse_plan(lines)
    return format_output(primitive_actions, task_methods)

def parse_plan(raw_plan_lines):
    primitive_actions = []
    task_method_mappings = []
    reading_primitives = True

    for line in raw_plan_lines:
        stripped = line.strip()
        if not stripped:
            continue

        if stripped.startswith("root"):
            reading_primitives = False
            continue

        if reading_primitives:
            parts = stripped.split()
            if parts and parts[0].isdigit():
                action_id = parts[0]
                action = " ".join(parts[1:])
                primitive_actions.append((action_id,action))
        else:
            parts = stripped.split("->")
            if len(parts) == 2:
                task_id = parts[0].strip().split()[0]
                task = ' '.join(parts[0].strip().split()[1:])
                method = parts[1].strip()  # Only keep method name
                task_method_mappings.append([task_id, task, method])

    return primitive_actions, task_method_mappings

def format_output(primitive_actions, task_method_mappings):
    output = []
    output.append("Step |   ID   | Primitive Action")
    output.append("---------------------------------------")
    for i, act in enumerate(primitive_actions, 1):
        output.append(f"{i:^5} | {act[0]:^6} | {act[1]}")

    output.append("\nStrategies used:")
    output.append(f"Task ID |    Task     ->     Method")
    output.append("---------------------------------------")
    max_width = max(len(str(item[1])) for item in task_method_mappings)
    for task_id, task, method in task_method_mappings:
        output.append(f"{task_id:^7} | {task:<{max_width}} -> {method}")
    return '\n'.join(output)