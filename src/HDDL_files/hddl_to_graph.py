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
                        if ":ordered-subtasks" in method_body:
                                subtask_start = method_body.find(":ordered-subtasks")
                                subtask_block = method_body[subtask_start:]
                                subtask_matches = re.findall(r'\((\S+)', subtask_block)
                                if "and" in subtask_matches:
                                        subtask_matches.remove("and") #remove 'and' since it is not a subtask
                                print("subtask matches for method_body {}:".format(method_body[9:19]), subtask_matches)
                                for ord, sub in enumerate(subtask_matches):
                                        if sub in self.tasks or sub in self.actions:
                                                self.graph.add_edge(method_name, sub, priority=ord)
                        elif ":subtasks" in method_body:
                                subtask_start = method_body.find(":subtasks")
                                subtask_block = method_body[subtask_start:]
                                subtasks_and_orders = extract_priority_from_subtasks_and_ordering(subtask_block)
                                for (sub, ord) in subtasks_and_orders:
                                        if sub in self.tasks or sub in self.actions:
                                                self.graph.add_edge(method_name, sub, priority=ord)
                        # 


def extract_priority_from_subtasks_and_ordering(block):
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

if __name__ == "__main__":
        domain_path = input("Please provide domain file directory, if want to use default file, click enter.")
        if domain_path == '':
                domain_path = "./test_input_modify_overcooked.hddl"  # Replace with your domain file
        elif not os.path.exists(domain_path):
                print("Domain file {} does NOT exist ... --> use default test domain instead")
                domain_path = "./test_input_modify_overcooked.hddl"
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
        has_cycle = not nx.is_directed_acyclic_graph(graph)
        print("CYCLE detected? ", has_cycle)
        if has_cycle:
                cycle = nx.find_cycle(graph, orientation='original')
                print("Cycle found: \n  ", cycle)

                all_cycles = list(nx.simple_cycles(graph))
                # Filter for cycles with more than 2 nodes
                long_cycles = [cycle for cycle in all_cycles if len(cycle) > 2]
                print("Cycles with >2 nodes:", long_cycles)

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
                label_x = x0 + (x1 - x0)*(0.1 + d['priority']*0.05)
                label_y = y0 + (y1 - y0)*(0.1 + d['priority']*0.05)
                label = f"[{d['priority']}]"
                plt.text(label_x, label_y, label, fontsize=15, color='red', fontweight='bold')
        plt.title('HDDL Operator Graph')
        # Save the figure
        save_dir = domain_path.replace('.hddl','_operators_graph.png')
        plt.savefig(save_dir, format="png", dpi=300, bbox_inches="tight")
        plt.show()

