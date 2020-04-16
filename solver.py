import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys

import matplotlib.pyplot as plt

def solve(G):
    """
    Args:
        G: networkx.Graph

    Returns:
        T: networkx.Graph
    """

    # TODO: your code here!

    mst = nx.minimum_spanning_tree(G)
    #print(sorted(mst.edges(data=True)))
    return mst

    # pass

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
    write_output_file(T, 'out/'+path[:len(path)-3]+'.out')
    #write_output_file(T, 'out/test.out')
