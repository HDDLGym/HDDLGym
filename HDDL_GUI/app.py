# app.py
from flask import Flask, render_template, request, jsonify
import os
import hddl_to_graph
# import CAI_hddl
import tools_hddl
from generate_htn_image import parse_plan, parse_plan_render
import networkx as nx
import LLM
import copy

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC'], exist_ok=True)
# GLOBAL DOMAIN = None
# GLOBAL PROBLEM = None
DOMAIN_TEXT = ""
PROBLEM_TEXT = ""
ORIGINAL_DOMAIN_TEXT = ""
def nx_to_cytoscape_elements(G: nx.MultiDiGraph):
    # nodes = [{'data': {'id': str(n), 'label': str(n)}} for n in G.nodes()]
    nodes = []
    for n, d in G.nodes(data=True):
            if d['type'] == 'task':
                    color = 'darkblue'
                    shape = 'box'
                    layer = 0
            elif d['type'] == 'method':
                    color = 'darkgreen'
                    shape = 'ellipse'
                    layer = 1
            elif d['type'] == 'action':
                    color = 'black'
                    shape = 'box3d'
                    layer = 2
            elif d['type'] == 'root':
                    color = 'red'
                    shape = 'dot'
                    layer = 0
            nodes.append({'data': {'id': str(n), 'label': str(n), 'color': color, 'type': d['type'], 'shape': shape, 'layer': layer}})
    edges = []
    for u, v, key, data in G.edges(keys=True, data=True):
            # edge_id = f"{u}-{v}-{key}"
            if 'priority' in data.keys():
                label = data['priority'] if data['priority'] not in (-1, None) else ""
            else: 
                label = ""
            edge = {
                    'data': {
                            # 'id': edge_id,
                            'source': str(u),
                            'target': str(v),
                            # 'priority': data['priority'],  # default to 0
                            'label': label  # optional: show on edge
                    }
            }
            # # Optionally add attributes like weight, label, etc.
            # edge['data'].update({k: v for k, v in data.items()})
            edges.append(edge)

    return nodes+ edges

def digraph_to_cytoscape_elements(G: nx.DiGraph):
    """
    Converts a directed graph (nx.DiGraph) into Cytoscape-compatible elements.

    Args:
        G (nx.DiGraph): Input directed graph.

    Returns:
        list: Cytoscape-compatible elements (nodes and edges).
    """
    # Process nodes
    nodes = []
    for n, d in G.nodes(data=True):
            if d['type'] == 'task':
                    color = 'darkblue'
                    shape = 'roundrectangle'
                    
            elif d['type'] == 'method':
                    color = 'darkgreen'
                    shape = 'ellipse'
                    
            elif d['type'] == 'action':
                    color = 'black'
                    shape = 'rectangle'
                    
            elif d['type'] == 'root':
                    color = 'red'
                    shape = 'diamond'
            nodes.append({'data': {'id': str(n), 'label': d['label'], 'color': color, 'type': d['type'], 'shape': shape, 'step': d['step']}})
    edges = []
    for u, v, data in G.edges(data=True):
            edge = {
                    'data': {
                            # 'id': edge_id,
                            'source': u,
                            'target': v,
                            # 'priority': data['priority'],  # default to 0
                    }
            }
            # # Optionally add attributes like weight, label, etc.
            # edge['data'].update({k: v for k, v in data.items()})
            edges.append(edge)

    return nodes+ edges

def plan_with_hddl_planner(return_format_version=True):
    """Plan with the HDDL planner"""
    global DOMAIN_PATH, PROBLEM_PATH, DOMAIN_TEXT, PROBLEM_TEXT

    # with open(DOMAIN_PATH, "r") as f:
    #     DOMAIN_TEXT = f.read()
    # with open(PROBLEM_PATH, "r") as f_prob:
    #     PROBLEM_TEXT = f_prob.read()
    success, info = tools_hddl.verifyMethodEncoding(DOMAIN_TEXT, PROBLEM_TEXT, new_methods_str="",current_path="./", debug = False, return_format_version=return_format_version)
    if success:
        return info
    else:
        return "Failed to plan:\n" + str(info)

# def run_planner():
#     '''
#     what to do when the user clicks the Plan button
#     '''
#     print("Clicked Plan button --> Perform planning with the system HDDL planner")
#     text = ''
#     digraph_diagram = dict()
#     result = plan_with_hddl_planner(return_format_version=False)
#     if "Failed to plan" in result:
#         text = result
#         return {'text': text, 'elements': []}
#     plan_time_str = result.splitlines()[-1]
#     raw_plan = "\n".join(result.splitlines()[:-1])
#     formated_plan_text = tools_hddl.format_lilotane_plan(raw_plan)
#     text = formated_plan_text+ '\n' + plan_time_str
#     #Generate the plan diagram:
#     digraph_diagram = parse_plan(raw_plan)
#     graph_data = digraph_to_cytoscape_elements(digraph_diagram)

#     return {'text': text, 'elements': graph_data}

# Dummy placeholder for domain text display
@app.route('/load_domain', methods=['POST'])
def load_domain():
    global DOMAIN_PATH, DOMAIN_TEXT, ORIGINAL_DOMAIN_TEXT
    domain_file = request.files.get('domain')

    if domain_file:
        DOMAIN_PATH = os.path.join(app.config['UPLOAD_FOLDER'], domain_file.filename)
        domain_file.save(DOMAIN_PATH)
        with open(DOMAIN_PATH, 'r') as f:
            DOMAIN_TEXT = f.read()
        ORIGINAL_DOMAIN_TEXT = DOMAIN_TEXT
        # reset the LLM message history:
        LLM.clear_message_history()
        # #   Use LLM to interpret the problem, then show in the display frame:
        # prompt_input = [{"role": "user", "content":f"Describe the following domain from the HDDL files in a more human-interpretable language. \Domain: {DOMAIN_TEXT}"}]
        # print("Use LLM to interpret the domain...")
        # response = LLM.call_llm('',prompt_input, save_history=True, use_history=True) 
    else:
        DOMAIN_TEXT = "(No domain file provided)"
        ORIGINAL_DOMAIN_TEXT = ''
        # response=''

    return jsonify({'domain': DOMAIN_TEXT})#, 'text_output': response})

@app.route('/load_problem', methods=['POST'])
def load_problem():
    global PROBLEM_PATH, PROBLEM_TEXT
    problem_file = request.files.get('problem')

    if problem_file:
        PROBLEM_PATH = os.path.join(app.config['UPLOAD_FOLDER'], problem_file.filename)
        problem_file.save(PROBLEM_PATH)
        with open(PROBLEM_PATH, 'r') as f:
            PROBLEM_TEXT = f.read()
        # # Use LLM to interpret the problem, then show in the display frame:
        # prompt_input = [{"role": "user", "content":f"Describe the following problem from the HDDL files in a more human-interpretable language. \nProblem: {PROBLEM_TEXT}"}]
        # print("Use LLM to interpret the problem...")
        # response = LLM.call_llm('',prompt_input, save_history=True, use_history=True) 
        # print("LLM response:", response)
    else:
        PROBLEM_TEXT = "(No domain file provided)"
        # response = '(No domain file provided)'

    return jsonify({'problem': PROBLEM_TEXT})#, 'text_output': response})

# Simulate HTN viewer response from LLM and graph
@app.route('/view_htn', methods=['POST'])
def view_htn():
    global DOMAIN_TEXT, DOMAIN_PATH
    DOMAIN_TEXT = request.json.get('domainText', '')
    # print('domain text:',DOMAIN_TEXT)
    
    # # Use LLM to interpret the domain and problem, then show in the display frame:
    # prompt_input = [{"role": "user", "content":f"Describe the following domain and problem from the HDDL files in a more human-interpretable language. \nDomain: {DOMAIN_TEXT}\nProblem: {PROBLEM_TEXT}"}]
    # print("Interpreting the domain and problem...")
    # # Simulate LLM response
    # prompt_input.append({"role": "user", "content":f"Additional request from the user: {query}"})
    # response = LLM.call_llm('',prompt_input)  
    temp_domain_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_domain.hddl')
    with open(temp_domain_path, 'w') as f:
        f.write(DOMAIN_TEXT)
    parser = hddl_to_graph.HDDLParser(temp_domain_path)
    nxgraph = parser.parse()
    graph_data = nx_to_cytoscape_elements(nxgraph)
    graph_data = {'elements': graph_data}

    return jsonify({'graph': graph_data})

# Delete methods:
@app.route('/delete_methods', methods=['POST'])
def delete_methods():
    global DOMAIN_TEXT, ORIGINAL_DOMAIN_TEXT
    list_operators_to_delete = request.json.get('methods', [])
    DOMAIN_TEXT = request.json.get('domainText', '')
    delete_methods = []
    updated_domain = copy.copy(DOMAIN_TEXT)
    for operator in list_operators_to_delete:
        if ":method "+operator in DOMAIN_TEXT:
             updated_domain = tools_hddl.removeMethod(updated_domain, delete_method=operator)
             delete_methods.append(operator)
    DOMAIN_TEXT = updated_domain
        
    return jsonify({'deleted_methods': delete_methods, 'updated_domain': DOMAIN_TEXT})

# Ask LLM:
@app.route('/ask_llm', methods=['POST'])
def ask_LLM():
    response = ''
    query = request.json.get('query', '')
    if query:
        response = LLM.call_llm('',[{"role": "user", "content":query}],save_history=True, use_history=True)
    else:
        response = '(No query provided)'
    return jsonify({'response': response})

@app.route('/run_planner', methods=['POST'])
def run_planner():
    '''
    run HDDL Planner and return text and graph data
    '''
    text = ''
    digraph_diagram = dict()
    result = plan_with_hddl_planner(return_format_version=False)
    if "Failed to plan" in result:
        text = result
        return jsonify({'text': text, 'elements': []})
        # return jsonify({'text': text, 'diagram_path': None})

    plan_time_str = result.splitlines()[-1]
    raw_plan = "\n".join(result.splitlines()[:-1])
    formated_plan_text = tools_hddl.format_lilotane_plan(raw_plan)
    text = formated_plan_text+ '\n' + plan_time_str
    #Generate the plan diagram:
    digraph_diagram = parse_plan(raw_plan)
    graph_data = digraph_to_cytoscape_elements(digraph_diagram)
    # Render the diagram and save to file
    # diagram_filepath = os.path.join(app.config['STATIC'],'plan_diagram')
    # digraph_diagram.render(filename=diagram_filepath, format='png', cleanup=True)
    return jsonify({'text': text, 'elements':graph_data})# 'diagram_path': '/static/plan_diagram.png'})
    
@app.route('/add_method', methods=['POST'])
def add_method():
    '''
    Add a new method to the domain
    '''
    global DOMAIN_TEXT
    method_text = request.json.get('latest_response', '')
    domain_text = request.json.get('domain_text', '')
    if not method_text:
        return jsonify({'status': 'No method text provided', 'updated_domain_text': domain_text})
    
    # Here you would typically add the method to the domain file
    updated_domain_text = tools_hddl.updateDomain(domain_text, new_method=method_text)
    DOMAIN_TEXT = updated_domain_text
    
    return jsonify({'status': 'Method added successfully', 'updated_domain_text': updated_domain_text})


@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
