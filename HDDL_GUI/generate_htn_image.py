import re
from graphviz import Digraph
import networkx as nx

def parse_plan_render(plan_text: str) -> Digraph:
    lines = plan_text.strip().splitlines()
    data = {}
    order = None
    warnings = []

    dot = Digraph("HTN")
    dot.attr(overlap='false')
    dot.attr(splines='true')
    dot.attr(dpi='300')
    dot.attr(size='10,5')

    for i, line in enumerate(lines):
        l = line.strip()
        if not l:
            continue

        action_match = re.match(r"^(\d+)\s+([\w\s()\-]+)$", l)
        method_match = re.match(r"^(\d+)\s+([\w\s()\-]+?)\s+->\s+([\w\-]+)\s*([\d\s]*)$", l)
        root_match = re.match(r"^root\s*([\d\s]*)$", l)
        step_count = 0
        if action_match:
            step_count += 1
            idx, label = action_match.groups()
            label = str(step_count) + ". " + label.strip()
            print("label:", label)
            data[idx] = (f'{idx}', label, 'box3d', [])
        elif method_match:
            idx, task, method, children = method_match.groups()
            child_list = children.strip().split() if children.strip() else []
            data[idx] = (f'{idx}', task, 'box', child_list)
            data[f"m{idx}"] = (f"m{idx}", method, 'ellipse', [])
        elif root_match:
            order = root_match.group(1).strip().split()
        else:
            warnings.append(f"Ignoring line {i+1}: {l}")

    if order is None:
        raise ValueError("Expected root")

    dot.node("root", shape="point")
    for o in order:
        dot.edge("root", o)

    queue = list(order)
    visited = set()
    # print(f"Queue: {queue}")

    while queue:
        current = queue.pop(0)
        # print(f"Current: {current}")
        if current in visited or current not in data:
            continue
        visited.add(current)

        node_id, label, shape, children = data[current]
        dot.node(node_id, label=label, shape=shape)

        for child in children:
            # dot.edge(node_id, child)
            queue.append(child)

        # If it's a method, add a separate node and edge to show method decomposition
        if current in data and current.startswith("m"):
            continue
        if f"m{current}" in data:
            method_id, method_label, _, _ = data[f"m{current}"]
            dot.node(method_id, label=method_label)
            dot.edge(current, method_id)
            for child in data[current][3]:
                dot.edge(method_id, child)
    # print("graph created", dot.source)
    return dot

def parse_plan(plan_text: str) -> nx.DiGraph:
    """
    Parses a plan text and constructs a NetworkX directed graph (nx.DiGraph).

    Args:
        plan_text (str): The plan text to parse.

    Returns:
        nx.DiGraph: A directed graph representing the HTN.
    """
    lines = plan_text.strip().splitlines()
    data = {}
    order = None
    warnings = []

    # Create a NetworkX directed graph
    graph = nx.DiGraph()
    step_count = 0
    for i, line in enumerate(lines):
        l = line.strip()
        if not l:
            continue

        action_match = re.match(r"^(\d+)\s+([\w\s()\-]+)$", l)
        method_match = re.match(r"^(\d+)\s+([\w\s()\-]+?)\s+->\s+([\w\-]+)\s*([\d\s]*)$", l)
        root_match = re.match(r"^root\s*([\d\s]*)$", l)
        
        if action_match:
            step_count += 1
            idx, label = action_match.groups()
            label = str(step_count) + ". " + label.strip()
            # print("label:", label)
            data[idx] = (f'{idx}', label, 'action', [], step_count)
        elif method_match:
            idx, task, method, children = method_match.groups()
            child_list = children.strip().split() if children.strip() else []
            data[idx] = (f'{idx}', task, 'task', child_list, 0)
            data[f"m{idx}"] = (f"m{idx}", method, 'method', [], 0)
        elif root_match:
            order = root_match.group(1).strip().split()
        else:
            warnings.append(f"Ignoring line {i+1}: {l}")

    if order is None:
        raise ValueError("Expected root")

    # Add the root node
    graph.add_node("root", label="root", shape="point", type="root", step=0)

    # Add edges from root to the initial tasks/methods
    for o in order:
        graph.add_edge("root", o)

    queue = list(order)
    visited = set()

    while queue:
        current = queue.pop(0)
        if current in visited or current not in data:
            continue
        visited.add(current)

        node_id, label, node_type, children, step = data[current]
        graph.add_node(node_id, label=label, type=node_type, step=step)

        for child in children:
            # graph.add_edge(node_id, child)
            queue.append(child)

        # If it's a method, add a separate node and edge to show method decomposition
        if current.startswith("m"):
            continue
        if f"m{current}" in data:
            method_id, method_label, _, _, step= data[f"m{current}"]
            graph.add_node(method_id, label=method_label, type="method", step=step)
            # print(f"Adding edge from {current} to method {method_id}")
            graph.add_edge(current, method_id)
            for child in data[current][3]:
                graph.add_edge(method_id, child)

    # Assign step values to tasks and methods based on the maximum step of their linked actions
    for node in reversed(list(nx.topological_sort(graph))):  # Process nodes in reverse topological order
        if graph.nodes[node]['type'] in {'task', 'method'}:
            max_step = graph.nodes[node].get('step', 0)
            for _, child in graph.out_edges(node):
                max_step = max(max_step, graph.nodes[child].get('step', 0))
            graph.nodes[node]['step'] = max_step
            # graph.nodes[node]['label'] += f" (Step: {max_step})"
    

    # Print warnings if any
    if warnings:
        print("\n".join(warnings))

    return graph


# Example usage
if __name__ == "__main__":
    with open("plan.txt", "r") as f:
        plan_content = f.read()

    match = re.search(r"==>\n([^<]*)", plan_content, re.DOTALL)
    input_text = match.group(1).strip() if match else plan_content

    dot_graph = parse_plan(input_text)
    dot_graph.render("htn_plan", format="png", cleanup=True)
    print("Saved as htn_plan.png")
