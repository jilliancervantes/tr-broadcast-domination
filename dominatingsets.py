import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import itertools
import colorsys

# Set numpy print options to display the full array (without truncation).
np.set_printoptions(threshold=sys.maxsize)


def construct_graph():
    # Create an empty undirected graph.
    construction = nx.Graph()
    i = 0

    # Adding nodes and edges for horizontal connections
    for y in range(m + 1):
        for x in range(n):
            construction.add_node(i, pos=(1 + 3 * x, 3 * y))
            construction.add_node(i + 1, pos=(2 + 3 * x, 3 * y))
            construction.add_edge(i, i + 1)
            i += 2

    # Adding nodes and edges for vertical connections
    j = i
    for y in range(m):
        for x in range(n + 1):
            construction.add_node(j, pos=(3 * x, 1 + 3 * y))
            construction.add_node(j + 1, pos=(3 * x, 2 + 3 * y))
            construction.add_edge(j, j + 1)
            j += 2

    # Adding diagonal edges
    row_start = 2 * n * (m + 1)  # Start of diagonal connections
    i = 1
    for y in range(m):
        for x in range(n):
            construction.add_edge(row_start + 2 * (x + 1), i)
            construction.add_edge(row_start + 2 * x, i - 1)
            construction.add_edge(row_start + 1 + 2 * x, i + 2 * n - 1)
            construction.add_edge(row_start + 1 + 2 * (x + 1), i + 2 * n)
            i += 2
        row_start += 2 * (n + 1)

    # Initialize node attributes: 'tower' is False, 'reception' is 0.
    nx.set_node_attributes(construction, False, 'tower')
    nx.set_node_attributes(construction, 0, 'reception')

    return construction


def display(construction):
    # Prepare the graph layout and visualization settings
    pos_list = nx.get_node_attributes(construction, "pos")
    colors, legend = get_colors_and_legend()
    sizes = get_node_sizes()

    fig, ax = plt.subplots()
    pos = pos_list
    # Draw nodes with color based on 'reception' and size based on 'tower' attribute
    nodes = nx.draw_networkx_nodes(construction, pos=pos, ax=ax,
                                   node_color=[colors[node[1]['reception']] for node in construction.nodes(data=True)],
                                   node_size=[sizes[node[1]['tower']] for node in construction.nodes(data=True)])
    nx.draw_networkx_edges(construction, pos=pos, ax=ax)
    nx.draw_networkx_labels(construction, pos=pos, ax=ax)  # Display node labels

    # Add a legend to the plot
    plt.legend(handles=legend, loc='upper right')

    # Annotation for hover effect
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        # Update annotation text based on the node being hovered over
        node = ind["ind"][0]
        xy = pos[node]
        annot.xy = xy
        node_attr = {'node': node}
        node_attr.update(construction.nodes[node])
        text = '\n'.join(f'{k}: {v}' for k, v in node_attr.items())
        annot.set_text(text)

    def hover(event):
        # Display annotation when hovering over a node
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = nodes.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)

    # Show the plot
    plt.show()


def dominating_set_search(graph):
    # Search for a dominating set in the graph
    can_dominate = False

    # Iterate through all possible subsets of nodes
    for subset in itertools.combinations(list(graph.nodes), (n * (m + 1) + m)):
        if nx.is_dominating_set(graph, subset):
            can_dominate = True
            print(f"{subset} DOMINATES GRAPH")
            break

    if can_dominate:
        print("Dominating set found")


def coordination_sequence(t):
    # Calculate the coordination sequence based on the input t
    t_index = t - 1
    if t_index % 3 == 0:
        count = 8 * (t_index // 3)
    elif t_index % 3 == 1:
        count = 8 * (t_index // 3) + 3
    elif t_index % 3 == 2:
        count = 8 * (t_index // 3) + 5
    return count


def reach(t):
    # Calculate the reach for t
    count = coordination_sequence(t)
    for i in range(2, t):
        count += coordination_sequence(i)
    return count + 1  # Add 1 for the dominating vertex itself


def place_towers(graph, vertices):
    # Place towers on specified vertices in the graph
    tower_dict = dict.fromkeys(vertices, True)
    nx.set_node_attributes(graph, tower_dict, 'tower')
    return


def get_tower_list(graph):
    # Retrieve the list of vertices where towers are placed
    tower_attribute_dict = nx.get_node_attributes(graph, 'tower')
    towers = [node for node, value in tower_attribute_dict.items() if value]
    return sorted(towers)


def calculate_reception(graph, t):
    # Calculate reception for each node based on the towers' placement
    towers = get_tower_list(graph)
    for tower in towers:
        nodes_in_reach = nx.single_source_shortest_path_length(graph, source=tower, cutoff=t - 1)
        print(f'Nodes in reach of {tower}: ', nodes_in_reach.keys())
        for node in nodes_in_reach:
            distance_to_tower = nx.shortest_path_length(graph, source=node, target=tower)
            reception_matrix[node, towers.index(tower)] = t - distance_to_tower
            current_reception = nx.get_node_attributes(graph, 'reception')[node]
            nx.set_node_attributes(graph, {node: current_reception + (t - distance_to_tower)}, 'reception')
    return


def get_colors_and_legend():
    # Generate a color map for nodes based on their reception values and create a legend
    max_reception = 0
    min_reception = 0
    try:
        max_reception = int(np.max(np.sum(reception_matrix, axis=1)))
        print(f'Maximum reception is {max_reception}.')
        min_reception = int(np.min(np.sum(reception_matrix, axis=1)))
        print(f'Minimum reception is {min_reception}.')
    except ValueError:
        max_reception = 0
        min_reception = 0
        print('Reception matrix is empty. Maximum and minimum reception are both 0.')
    except NameError:
        max_reception = 0
        min_reception = 0
        print('Reception matrix is not defined. Maximum and minimum reception are both 0.')
    finally:
        color_state_map = {}
        legend_elements = []
        for i in range(min_reception, max_reception + 1):
            h = i / (max_reception + 1)  # Vary hue between 0 and 1
            s = 1.0  # Full saturation
            v = 1.0  # Full brightness
            color = colorsys.hsv_to_rgb(h, s, v)
            color_state_map[i] = color
            legend_elements.append(Line2D([0], [0], marker='o', color='w', label=f'r(v) = {i}', markerfacecolor=color,
                                          markersize=10))
    return color_state_map, legend_elements


def get_node_sizes():
    # Define node sizes based on whether a tower is placed on them
    node_sizes = {True: 700, False: 300}
    return node_sizes


def is_t_r_dominating_set(reception_mat, r):
    # Check if the towers form a (t,r) broadcast dominating set
    print(reception_mat)
    sum_reception = np.sum(reception_mat, axis=1)
    if np.all(sum_reception >= r):
        return True
    else:
        not_dominated = np.where(sum_reception < r)[0]
        print('The following vertices do not have enough reception:', not_dominated)
        return False


def confirm_domination_number(graph, proposed_number):
    # Confirm if the proposed number of towers forms a dominating set, ONLY FOR (t,r) = (2,1)
    set_can_dominate = False

    vertex_subsets = []
    for comb in itertools.combinations((list(graph.nodes)), (proposed_number - 1)):
        vertex_subsets.append(comb)

    for vertex_subset in vertex_subsets:
        if nx.is_dominating_set(graph, vertex_subset):
            set_can_dominate = True
            print(f"{vertex_subset} dominates the graph!")

    if set_can_dominate:
        print(f"Dominating set(s) found")
    return


# Parameters for graph construction
m, n = 5, 6
transmission_strength = 4
reception_threshold = 2

# Construct the graph
my_graph = construct_graph()

# List of vertices where towers will be placed
tower_list = [0, 9, 32, 116, 82, 30]
place_towers(my_graph, tower_list)

# Initialize the reception matrix
reception_matrix = np.zeros((2 * m + n * (4 * m + 2), len(tower_list)))

# Calculate reception for the graph
calculate_reception(my_graph, t=transmission_strength)
reception_dict = nx.get_node_attributes(my_graph, 'reception')

# Display the graph with towers and reception levels
display(my_graph)

# Check if the graph is a (t,r) broadcast dominating set and print the result
print(is_t_r_dominating_set(reception_matrix, reception_threshold))
print(reception_dict)
