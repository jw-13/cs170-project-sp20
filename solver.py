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

    #draw graph:
    """
    pos=nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G,"weight")
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.show()
    """

    mst = nx.minimum_spanning_tree(G)

    #draw tree:
    """
    tree_pos = nx.spring_layout(mst)
    nx.draw_networkx(mst, tree_pos)
    plt.show()
    """

    return mst

# Here's an example of how to run your solver.

# Usage: python3 solver.py test.in

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