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

    #random graph:
    #G = nx.connected_watts_strogatz_graph(n=25, k=4, p=0.8)

    # drawing the graph with weights: spring_layout same length edges / circular_layout easier to see degrees
    pos=nx.circular_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G,"weight")
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    plt.show()

    print([e for e in G.edges])

    mst = nx.minimum_spanning_tree(G)
    #print([i for i in mst.edges])

    # drawing tree:
    #tree_pos = nx.spring_layout(mst)
    #nx.draw_networkx(mst, tree_pos)
    #plt.show()
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
    #write_output_file(T, 'out/'+path[:len(path)-3]+'.out')
    #write_output_file(T, 'out/test.out')
