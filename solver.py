import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys

import matplotlib.pyplot as plt
import re #regex

def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        T: networkx.Graph
    """
    # TODO: your code here!

    #to draw graph
    plt.figure(figsize=(8,8))
    pos=nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G,"weight")
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)

    v_descending_degree = sorted([n for n, d in G.degree()], reverse=True, key=G.degree()) #vertices sorted by degree
    v_ascending_degree = sorted([n for n, d in G.degree()], reverse=False, key=G.degree())

    v_deg_1 = [v for v in v_ascending_degree if G.degree[v]==1] #leaves
    max_v = max(v_descending_degree) #vertex with max degree

    T = nx.minimum_spanning_tree(G)
    result = T

    for v in v_deg_1:
        copy_result = result.copy()
        copy_result.remove_node(v)
        if average_pairwise_distance(copy_result) <= average_pairwise_distance(result):
            result = copy_result

    covered_set = {}
    print("covered" , covered_set)

    #print(len(G.__getitem__(0))) #number of neighbors in G
    #len(G.__getitem__(0)) - len(T.__getitem__(0))
    #print(G.__getitem__(1)) #get list of neighbors

    #to draw tree in the same plot as G, uncomment plt.show()
    nx.draw_networkx_edges(G,pos,
        edgelist=[e for e in T.edges()],
        width=5,alpha=0.5,edge_color='r')
    #plt.show()

    #is_valid_network(T) to check if it works
    #total_pairwise_distance = average_pairwise_distance(T) * (len(T) * (len(T) - 1))
    return result

# Here's an example of how to run your solver.
# Usage: python3 solver.py inputs/small-302.in

if __name__ == '__main__':
    assert len(sys.argv) == 2
    path = sys.argv[1]

    G = read_input_file(path)

    """
    if you want to plot the graph
    nx.draw(G, with_labels=True, font_weight='bold')
    plt.show()
    """

    T = solve(G)
    assert is_valid_network(G, T)
    print("Average  pairwise distance: {}".format(average_pairwise_distance(T)))
    path_string = re.split('[/.]', path)
    write_output_file(T, 'output/'+path_string[1]+'.out')