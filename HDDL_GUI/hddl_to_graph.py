import re
import networkx as nx
import matplotlib.pyplot as plt
import os
from typing import Dict, List

class HDDLParser:
        def __init__(self, domain_file: str):
                with open(domain_file, 'r') as f:
                        self.domain_str = f.read()

                self.graph = nx.MultiDiGraph()
                self.tasks = {}
                self.methods = {}
                self.actions = {}

        def parse(self):
                self._parse_tasks()
                self._parse_methods()
                self._parse_actions()
                self._build_graph()
                return self.graph
        
        def get_operator_graph(self, operator_name: str) -> nx.MultiDiGraph:
                """ Generate a subgraph for a specific operator (task, method, or action) and its dependencies.
                input:
                operator_name: The name of the operator (task, method, or action) to generate the subgraph for.
                """
                operator_graph = nx.MultiDiGraph()
                # Add all nodes and edges related to the operator
                for node in self.graph.nodes:
                        if node == operator_name or self.graph.has_edge(operator_name, node):# or self.graph.has_edge(node, operator_name):
                                operator_graph.add_node(node, type=self.graph.nodes[node]['type'])
                # Add edges connecting the operator to its dependencies:
                operator_graph.add_edges_from(self.graph.edges(operator_name, keys=True))
                return operator_graph
        
        def generate_operator_graph_image(self, operator_name: str, save_dir=None) -> str:
                """ Generate and save a subgraph for a specific operator (task, method, or action) and its dependencies.
                input:
                operator_name: The name of the operator (task, method, or action) to generate the subgraph for.
                save_dir: The directory to save the graph image. If None, saves in the same directory as the domain file.
                """
                operator_graph = self.get_operator_graph(operator_name)
                plt.figure(figsize=(10, 10))
                pos = nx.shell_layout(operator_graph)  # You can try other layouts like nx.shell_layout
                node_colors = []
                for _, data in operator_graph.nodes(data=True):
                        if data['type'] == 'task':
                                node_colors.append('skyblue')
                        elif data['type'] == 'method':
                                node_colors.append('lightgreen')
                        elif data['type'] == 'action':
                                node_colors.append('salmon')
                        else:
                                node_colors.append('gray')
                nx.draw(operator_graph, pos, with_labels=True, node_color=node_colors, node_size=12000, font_size=10, font_weight='bold', arrows=True)
                offset = 0.05  # vertical offset to avoid overlap
                for i, (u, v, k, d) in enumerate(operator_graph.edges(keys=True, data=True)):
                        if d['priority'] == -1:
                                continue
                        x0, y0 = pos[u]
                        x1, y1 = pos[v]
                        label_x = x0 + (x1 - x0)*(0.1 + d['priority']*0.01)
                        label_y = y0 + (y1 - y0)*(0.1 + d['priority']*0.01)
                        label = f"[{d['priority']}]"
                        plt.text(label_x, label_y, label, fontsize=15, color='red', fontweight='bold')
                plt.title(f'HDDL Operator Graph for {operator_name}')
                # Save the figure
                if save_dir is None:
                        save_dir = os.path.join(os.path.dirname(self.domain_str), f'{operator_name}_operators_graph.png')
                plt.savefig(save_dir, format="png", dpi=300, bbox_inches="tight")
                plt.close()
                return save_dir

        def _parse_tasks(self):
                
                # task_pattern = re.compile(r'\(:task (.*?)\)', re.DOTALL)
                # for match in task_pattern.findall(self.domain_str):
                task_blocks = self.extract_blocks(self.domain_str, ":task")
                for match in task_blocks:
                        header = match.strip().splitlines()[0]
                        name = header.split()[1]  # e.g., (:task task_name ...)
                        self.tasks[name] = match
                        self.graph.add_node(name, type='task')

        def extract_blocks(self, text, keyword=":method"):
                blocks = []
                start = 0
                while True:
                        start = text.find(f"({keyword}", start)
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
        def _parse_methods(self):
        
                method_blocks = self.extract_blocks(self.domain_str, ":method")
                for block in method_blocks:
                        header = block.strip().splitlines()[0]
                        name = header.split()[1]  # e.g., (:method method_name ...)
                        self.methods[name] = block
                        self.graph.add_node(name, type='method')

        def _parse_actions(self):
                # action_pattern = re.compile(r'\(:action (.*?)\)', re.DOTALL)
                action_blocks = self.extract_blocks(self.domain_str, ":action")
                # for match in action_pattern.findall(self.domain_str):
                for block in action_blocks:
                        header = block.strip().splitlines()[0]
                        name = header.split()[1]  # e.g., (:action action_name ...)
                        self.actions[name] = block
                        self.graph.add_node(name, type='action')

        def _build_graph(self):
                for method_name, method_body in self.methods.items():
                        # Find task this method decomposes
                        # print("method_body of method name {} is: {}".format(method_name, method_body))
                        decomposes_match = re.search(r':task \((\S+)', method_body)
                        # print("decomposes_match of method name {} is: {}".format(method_name, decomposes_match))
                        if decomposes_match:
                                task_name = decomposes_match.group(1)
                                if task_name in self.tasks:
                                        self.graph.add_edge(task_name, method_name, priority=-1)

                        # Find subtasks or actions used in the method body
                        # Also consider ":ordered-subtasks" and ":subtasks" blocks
                        subtask_start = method_body.find(":subtasks")
                        if subtask_start != -1:
                                subtask_block = method_body[subtask_start:]
                                subtask_matches = re.findall(r'\((\S+)', subtask_block)
                                subtasks_and_orders = extract_priority_from_subtasks_and_ordering(subtask_block)
                                for (sub, ord) in subtasks_and_orders:
                                        if sub in self.tasks or sub in self.actions:
                                                self.graph.add_edge(method_name, sub, priority=ord)
                                # for sub in subtask_matches:
                                #         if sub in self.tasks or sub in self.actions:
                                #                 self.graph.add_edge(method_name, sub)
                        elif ":ordered-subtasks" in method_body:
                                subtask_start = method_body.find(":ordered-subtasks")
                                subtask_block = method_body[subtask_start:] if subtask_start != -1 else ""
                                subtasks_and_orders = extract_priority_from_subtasks_and_ordering(subtask_block)
                                for (sub, ord) in subtasks_and_orders:
                                        if sub in self.tasks or sub in self.actions:
                                                self.graph.add_edge(method_name, sub, priority=ord)
                        elif ":ordered-tasks" in method_body:
                                subtask_start = method_body.find(":ordered-tasks")
                                subtask_block = method_body[subtask_start:] if subtask_start != -1 else ""
                                subtasks_and_orders = extract_priority_from_subtasks_and_ordering(subtask_block)
                                for (sub, ord) in subtasks_and_orders:
                                        if sub in self.tasks or sub in self.actions:
                                                self.graph.add_edge(method_name, sub, priority=ord)

                                
def extract_ordered_subtasks(block):
        """
        Extracts the content of :ordered-subtasks while ensuring parentheses match correctly.
        Args:
                block (str): The block of text containing :ordered-subtasks.
        Returns:
                List[str]: A list of subtask strings.
        """
        # Find the start of :ordered-subtasks
        match = re.search(r':ordered-subtasks\s*\(', block)
        if not match:
                return []

        # Extract the substring starting from :ordered-subtasks
        start_index = match.end() - 1  # Position after the opening parenthesis
        end_index = find_matching_parenthesis(block, start_index)

        if end_index == -1:
                raise ValueError("Unmatched parentheses in :ordered-subtasks block.")

        # Extract the content between the matched parentheses
        content = block[start_index + 1:end_index]
        
        # Extract individual subtasks enclosed in parentheses
        subtasks = re.findall(r'\((.*?)\)', content)
        return subtasks

def find_matching_parenthesis(text, start_index):
    """
    Finds the index of the matching closing parenthesis for the opening parenthesis at start_index.
    Args:
        text (str): The text to search.
        start_index (int): The index of the opening parenthesis.
    Returns:
        int: The index of the matching closing parenthesis, or -1 if unmatched.
    """
    stack = 0
    for i in range(start_index, len(text)):
        if text[i] == '(':
            stack += 1
        elif text[i] == ')':
            stack -= 1
            if stack == 0:
                return i
    return -1  # No matching closing parenthesis found

def extract_priority_from_subtasks_and_ordering(block):
        '''
        Extracts subtasks and their ordering from a block of text.
        Args:
            block (str): The block of text containing subtasks and ordering constraints.
        Returns:
                List[Tuple[str, int]]: A list of tuples where each tuple contains a subtask label and its priority index.
        '''
        print("block:", block)
        ordered_actions = []
        if ":ordered-" in block:
                
                # Add subtask strings and their order to the ordered_action list. 
                # Note that the order is determined by the order of appearance in the block.
                # block is in format: ":ordered-subtasks (and (subtask1) (subtask2) ...)" or ":ordered-subtasks (subtask)"
                # save the subtask and order in ordered_actions: [(subtask1, 0), (subtask2, 1), ...]
                # Match the format ":ordered-subtasks (and (subtask1) (subtask2) ...)" or ":ordered-subtasks (subtask)"
                # subtask_matches = re.findall(r':ordered-subtasks\s*\((?:and\s*)?([\s\S]+?)\)', block)
                subtask_matches = re.findall(r':ordered-subtasks\s*\((?:and\s*)?(.*?)\)', block)
                if not subtask_matches:
                        subtask_matches = re.findall(r':ordered-tasks\s*\((?:and\s*)?(.*?)\)', block)
                if subtask_matches:
                        print("subtask_matches: ", subtask_matches)
                        # Extract individual subtasks by splitting the matched string
                        subtasks = re.findall(r'\((.*?)\)', subtask_matches[0])
                        # Enumerate the subtasks and save them with their order
                        print("Subtasks: ", subtasks)
                        ordered_actions = [(subtask, i) for i, subtask in enumerate(subtasks)]
        
        elif ":subtasks" in block and ":ordering" in block:
                # Extract subtask labels and their actions
                subtask_matches = re.findall(r'\(\s*(task\d+)\s+\((\S+)', block)
                label_to_action = {label: action for label, action in subtask_matches}

                # Extract ordering constraints
                ordering_matches = re.findall(r'<\s+(task\d+)\s+(task\d+)', block)

                # Build dependency graph
                G = nx.DiGraph()
                for label in label_to_action:
                        G.add_node(label)
                for before, after in ordering_matches:
                        G.add_edge(before, after)

                # Topological sort gives execution order
                sorted_labels = list(nx.topological_sort(G))

                # Assign priority index based on topological order
                ordered_actions = [(label_to_action[label], i) for i, label in enumerate(sorted_labels)]
        
        return ordered_actions

def get_domain_graph(domain_path: str) -> nx.MultiDiGraph:
        parser = HDDLParser(domain_path)
        graph = parser.parse()
        return graph

def get_domain_graph_image_saved(domain_path: str, save_dir=None) -> nx.MultiDiGraph:
        parser = HDDLParser(domain_path)
        graph = parser.parse()
        plt.figure(figsize=(15, 10))  # width x height in inches
        pos = nx.shell_layout(graph)  # You can try other layouts like nx.shell_layout
        node_colors = []

        for _, data in graph.nodes(data=True):
                if data['type'] == 'task':
                        node_colors.append('yellow')
                elif data['type'] == 'method':
                        node_colors.append('lightgreen')
                elif data['type'] == 'action':
                        node_colors.append('lightgray')
                else:
                        node_colors.append('gray')

        nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=12000, font_size=18, font_weight='bold', arrows=True)
        offset = 0.05  # vertical offset to avoid overlap
        for i, (u, v, k, d) in enumerate(graph.edges(keys=True, data=True)):
                if d['priority'] == -1:
                        continue
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                # label_x = x0 + 0.25 * (x1 - x0)
                # label_y = y0 + 0.25 * (y1 - y0) + (i * offset)  # offset by edge index to avoid overlap
                label_x = x0 + (x1 - x0)*(0.15 + d['priority']*0.05)
                label_y = y0 + (y1 - y0)*(0.15 + d['priority']*0.05)
                label = f"[{d['priority']}]"
                plt.text(label_x, label_y, label, fontsize=15, color='red', fontweight='bold')
        plt.title('HDDL Operator Graph', fontsize=25)
        # Save the figure
        if save_dir is None:
                save_dir = domain_path.replace('.hddl','_operators_graph.png')
        plt.savefig(save_dir, format="png", dpi=300, bbox_inches="tight")
        return save_dir

def get_operator_graph(operator_name: str, domain_path=None) -> nx.MultiDiGraph:
        """ Generate a subgraph for a specific operator (task, method, or action) and its dependencies.
        input:
        operator_name: The name of the operator (task, method, or action) to generate the subgraph for.
        """
        domain_graph = HDDLParser(domain_path).parse()
        operator_graph = nx.MultiDiGraph()
        # Add all nodes and edges related to the operator
        operator_graph.add_node(operator_name, type=domain_graph.nodes[operator_name]['type'])
        for node in domain_graph.nodes:
                if domain_graph.has_edge(operator_name, node):# or self.graph.has_edge(node, operator_name):
                        operator_graph.add_node(node, type=domain_graph.nodes[node]['type'])
                        # priority = domain_graph.edges[operator_name][node]['priority']
                        print("operator_name: ", operator_name)
                        print("node: ", node)
                        for start, target, key, priority_data in domain_graph.edges(data=True, keys=True):
                                if start == operator_name and target == node:
                                        print("start: ", start)
                                        print("target: ", target)
                                        print("key: ", key)
                                        print("priority_data: ", priority_data)
                                        operator_graph.add_edge(operator_name, node, key=key, priority=priority_data['priority'])
                                        
                                # print("key: ", key)
                                # print("priority_data: ", priority_data)
                                # operator_graph.add_edge(operator_name, node, key=key, priority=priority_data)
                        # operator_graph.add_edge(operator_name, node, priority=domain_graph.edges[operator_name][node]['priority'])
        # Add edges connecting the operator to its dependencies:
        # operator_graph.add_edges_from(domain_graph.edges(operator_name, keys=True))
        print("operator_graph nodes: ", operator_graph.nodes(data=True))
        print("operator_graph edges: ", operator_graph.edges(keys=True, data=True))
        return operator_graph

# from networkx.drawing.nx_agraph import graphviz_layout
def get_operator_graph_image_saved(operator_name: str, domain_path=None, save_dir=None) -> str:
        """ Generate and save a subgraph for a specific operator (task, method, or action) and its dependencies.
        input: 
        operator_name: The name of the operator (task, method, or action) to generate the subgraph for.
        save_dir: The directory to save the graph image. If None, saves in the same directory as the domain file.
        """
        operator_graph = get_operator_graph(operator_name, domain_path)
        print("creating figure to store graph")
        plt.figure(figsize=(10, 5))
        pos = nx.shell_layout(operator_graph)  # You can try other layouts like nx.shell_layout
        # pos = nx.multipartite_layout(operator_graph, subset_key="layer")
        node_colors = []
        for _, data in operator_graph.nodes(data=True):
                if data['type'] == 'task':
                        node_colors.append('yellow')
                elif data['type'] == 'method':
                        node_colors.append('lightgreen')
                elif data['type'] == 'action':
                        node_colors.append('lightgray')
                else:
                        node_colors.append('gray')
        nx.draw(operator_graph, pos, with_labels=True, node_color=node_colors, node_size=2400, font_size=10, font_weight='bold', arrows=True)
        offset = 0.05  # vertical offset to avoid overlap
        for i, (u, v, k, d) in enumerate(operator_graph.edges(keys=True, data=True)):
                if d['priority'] == -1:
                        continue
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                label_x = x0 + (x1 - x0)*(0.1 + d['priority']*0.05)
                label_y = y0 + (y1 - y0)*(0.1 + d['priority']*0.05)
                label = f"[{d['priority']}]"
                plt.text(label_x, label_y, label, fontsize=12, color='red', fontweight='bold')
        plt.title(f'HDDL Operator Graph for {operator_name}')
        # Save the figure
        if save_dir is None:
                save_dir = os.path.join(os.path.dirname(domain_path), f'{operator_name}_operators_graph.png')
        plt.savefig(save_dir, format="png", dpi=300, bbox_inches="tight")
        print("returned save_dir: ", save_dir)
        return save_dir

def nx_to_cytoscape_elements(G: nx.MultiDiGraph):
        # nodes = [{'data': {'id': str(n), 'label': str(n)}} for n in G.nodes()]
        nodes = []
        for n, d in G.nodes(data=True):
                if d['type'] == 'task':
                        color = 'darkblue'
                elif d['type'] == 'method':
                        color = 'darkgreen'
                elif d['type'] == 'action':
                        color = 'black'
                nodes.append({'data': {'id': str(n), 'label': str(n), 'color': color, 'type': d['type']}})
        edges = []
        for u, v, key, data in G.edges(keys=True, data=True):
                # edge_id = f"{u}-{v}-{key}"
                label = data['priority'] if data['priority'] not in (-1, None) else ""
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


if __name__ == "__main__":
        domain_path = input("Please provide domain file directory, if want to use default file, click enter.")
        if domain_path == '':
                domain_path = "HDDL_env/zeno_domain.hddl"  # Replace with your domain file
        elif not os.path.exists(domain_path):
                print("Domain file {} does NOT exist ... --> use default test domain instead")
                domain_path = "domain.hddl"
        parser = HDDLParser(domain_path)
        graph = parser.parse()

        # Optionally, visualize or print the graph
        print("Nodes:", graph.nodes(data=True))
        print("Edges:", list(graph.edges))
        

        # Draw the graph with labels and node types
        plt.figure(figsize=(20, 20))  # width x height in inches

        pos = nx.shell_layout(graph)  # You can try other layouts like nx.shell_layout
        node_colors = []

        for _, data in graph.nodes(data=True):
                if data['type'] == 'task':
                        node_colors.append('skyblue')
                elif data['type'] == 'method':
                        node_colors.append('lightgreen')
                elif data['type'] == 'action':
                        node_colors.append('salmon')
                else:
                        node_colors.append('gray')

        nx.draw(graph, pos, with_labels=True, node_color=node_colors, node_size=12000, font_size=10, font_weight='bold', arrows=True)
        
        # edge_labels = nx.get_edge_attributes(graph, 'priority')
        # nx.draw_networkx_edge_labels(graph, pos, font_size=15, edge_labels=edge_labels,font_color='red', label_pos = 0.9)
        
        # edge_labels = {(u, v, k): f"priority={d['priority']}" for u, v, k, d in graph.edges(keys=True, data=True)}
        # # Draw edge labels
        # nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels, font_color='red', label_pos=0.3)

        offset = 0.05  # vertical offset to avoid overlap
        for i, (u, v, k, d) in enumerate(graph.edges(keys=True, data=True)):
                if d['priority'] == -1:
                        continue
                x0, y0 = pos[u]
                x1, y1 = pos[v]
                # label_x = x0 + 0.25 * (x1 - x0)
                # label_y = y0 + 0.25 * (y1 - y0) + (i * offset)  # offset by edge index to avoid overlap
                label_x = x0 + (x1 - x0)*(0.1 + d['priority']*0.01)
                label_y = y0 + (y1 - y0)*(0.1 + d['priority']*0.01)
                label = f"[{d['priority']}]"
                plt.text(label_x, label_y, label, fontsize=15, color='red', fontweight='bold')
        plt.title('HDDL Operator Graph')
        # Save the figure
        save_dir = domain_path.replace('.hddl','_operators_graph.png')
        plt.savefig(save_dir, format="png", dpi=300, bbox_inches="tight")
        plt.show()

