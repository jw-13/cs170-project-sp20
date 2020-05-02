import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys

import matplotlib.pyplot as plt
import re
from networkx.algorithms import tree

from heapq import heappop, heappush
from itertools import count

def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        T: networkx.Graph
    """

    result_edges = tree.mst.prim_mst_edges(G, True)
    edgelist = list(result_edges)

    str_edgelist = []
    for edge in edgelist:
        str_edgelist.append(str(str(edge[0])+" "+str(edge[1])+" "+str(edge[2]["weight"])))
    T = nx.parse_edgelist(str_edgelist, nodetype=int, data=(('weight',float),))

    result_tree = T

    v_descending_degree = sorted([n for n, d in G.degree()], reverse=True, key=G.degree()) #vertices sorted by degree
    max_v = max(v_descending_degree) #vertex with max degree

    curr_set = {max_v}
    N = len(v_descending_degree)
    i = 0
    while (i < N):
        new_set = curr_set
        new_set.update(G.__getitem__(i))
        if (len(new_set) < N):
            if (not set(G.__getitem__(i)).issubset(curr_set)):
                curr_set = curr_set.add(i)
        i += 1


    #print(len(G.__getitem__(0))) #number of neighbors in G
    #len(G.__getitem__(0)) - len(T.__getitem__(0))
    #print(G.__getitem__(1)) #get list of neighbors

    #is_valid_network(T) to check if it works
    #total_pairwise_distance = average_pairwise_distance(T) * (len(T) * (len(T) - 1))
    return result_tree


# prim's algo from networkx
#returns iterator over edges of mst
def prim_mst_edges(G, minimum, weight='weight',
                   keys=True, data=True, ignore_nan=False):
    """Iterate over edges of Prim's algorithm min/max spanning tree.

    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.

    minimum : bool (default: True)
        Find the minimum (True) or maximum (False) spanning tree.

    weight : string (default: 'weight')
        The name of the edge attribute holding the edge weights.

    keys : bool (default: True)
        If `G` is a multigraph, `keys` controls whether edge keys ar yielded.
        Otherwise `keys` is ignored.

    data : bool (default: True)
        Flag for whether to yield edge attribute dicts.
        If True, yield edges `(u, v, d)`, where `d` is the attribute dict.
        If False, yield edges `(u, v)`.

    ignore_nan : bool (default: False)
        If a NaN is found as an edge weight normally an exception is raised.
        If `ignore_nan is True` then that edge is ignored instead.

    """
    push = heappush
    pop = heappop

    nodes = set(G)
    c = count()

    while nodes:
        u = nodes.pop()
        frontier = []
        visited = {u}
        for v, d in G.adj[u].items():
            wt = d.get(weight, 1)
            push(frontier, (wt, next(c), u, v, d))
        while frontier:
            W, _, u, v, d = pop(frontier)
            if v in visited or v not in nodes:
                continue
            if data:
                yield u, v, d
            else:
                yield u, v
            # update frontier
            visited.add(v)
            nodes.discard(v)
            for w, d2 in G.adj[v].items():
                if w in visited:
                    continue
                new_weight = d2.get(weight, 1)
                push(frontier, (new_weight, next(c), v, w, d2))

if __name__ == '__main__':
    assert len(sys.argv) == 2

    #to run on all inputs: python3 solver.py all_inputs
    if sys.argv[1] == "all_inputs":
        """
        for i in range(1,304):
            path = 'inputs/small-'+str(i)+'.in'
            G = read_input_file(path)
            T = solve(G)
            assert is_valid_network(G, T)
            print(path + "Average  pairwise distance: {}".format(average_pairwise_distance(T)))
            path_string = re.split('[/.]', path)
            write_output_file(T, 'outputs/'+path_string[1]+'.out')
        for i in range(1,304):
            path = 'inputs/medium-'+str(i)+'.in'
            G = read_input_file(path)
            T = solve(G)
            assert is_valid_network(G, T)
            print(path + "Average  pairwise distance: {}".format(average_pairwise_distance(T)))
            path_string = re.split('[/.]', path)
            write_output_file(T, 'outputs/'+path_string[1]+'.out')
        for i in range(1,401):
            path = 'inputs/large-'+str(i)+'.in'
            G = read_input_file(path)
            T = solve(G)
            assert is_valid_network(G, T)
            print(path + "Average  pairwise distance: {}".format(average_pairwise_distance(T)))
            path_string = re.split('[/.]', path)
            write_output_file(T, 'outputs/'+path_string[1]+'.out')
        """
    else:
        path = sys.argv[1]
        G = read_input_file(path)
        T = solve(G)
        assert is_valid_network(G, T)
        print(path + "Average  pairwise distance: {}".format(average_pairwise_distance(T)))
        path_string = re.split('[/.]', path)
        write_output_file(T, 'outputs/'+path_string[1]+'.out')
