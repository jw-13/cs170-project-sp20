import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys

from random import choice
import re
from networkx.algorithms import tree
import numpy as np

from heapq import heappop, heappush
from itertools import count

def solve(G):
    """
    Args:
        G: networkx.Graph
    Returns:
        T: networkx.Graph
    """
    v_descending_degree = sorted([n for n, d in G.degree()], reverse=True, key=G.degree()) #vertices sorted by degree
    max_v = v_descending_degree[0]

    starting_v = max_v
    if (len(G.__getitem__(starting_v))) != 0:
        curr_min_avg = sum([v3['weight'] for v1,v2,v3 in G.edges.data() if v1==starting_v or v2==starting_v]) / len(G.__getitem__(0))
    else:
        curr_min_avg = 0

    for v in G.nodes:
        sum_incident_edges = sum([v3['weight'] for v1,v2,v3 in G.edges.data() if v1==v or v2==v])
        if len(G.__getitem__(v)) != 0:
            curr_avg = sum_incident_edges / len(G.__getitem__(v))
            if curr_avg < curr_min_avg:
                starting_v = v
                curr_min_avg = curr_avg

    v_descending_degree = sorted([n for n, d in G.degree()], reverse=True, key=G.degree()) #vertices sorted by degree
    max_v = v_descending_degree[0]
    if len(G.__getitem__(max_v)) == G.number_of_nodes()-1:
        T = nx.Graph()
        T.add_node(max_v)
        return T

    results = [0] * 5
    costs = [0] * 5
    T = nx.Graph()
    for i in range(0,5):
        starting_v = choice(list(G.nodes))
        print(starting_v)
        mst_edges = prim_mst_edges(G, starting_v)
        edgelist = list(mst_edges)
        str_edgelist = []
        for edge in edgelist:
            str_edgelist.append(str(str(edge[0])+" "+str(edge[1])+" "+str(edge[2]["weight"])))
        T = nx.parse_edgelist(str_edgelist, nodetype=int, data=(('weight',float),))

        v_ascending_degree = sorted([n for n, d in T.degree()], reverse=False, key=T.degree())
        v_deg_1 = [v for v in v_ascending_degree if T.degree[v]==1] #all leaves of tree

            #looking at all leaves
        for v in v_deg_1:
            copy_result = T.copy()
            copy_result.remove_node(v)
            if (copy_result.size() > 0) and nx.is_connected(copy_result):
                if average_pairwise_distance(copy_result) <= average_pairwise_distance(T):
                    T = copy_result
        results[i] = T
        costs[i] = average_pairwise_distance(T)
        print(costs)
    idx = np.argmin(costs)
    return results[idx]

    """
    mst_edges = prim_mst_edges(G, starting_v)
    edgelist = list(mst_edges)
    str_edgelist = []
    for edge in edgelist:
        str_edgelist.append(str(str(edge[0])+" "+str(edge[1])+" "+str(edge[2]["weight"])))
    T = nx.parse_edgelist(str_edgelist, nodetype=int, data=(('weight',float),))

    v_ascending_degree = sorted([n for n, d in T.degree()], reverse=False, key=T.degree())
    v_deg_1 = [v for v in v_ascending_degree if T.degree[v]==1] #all leaves of tree

    #looking at all leaves
    for v in v_deg_1:
        copy_result = T.copy()
        copy_result.remove_node(v)
        if (copy_result.size() > 0) and nx.is_connected(copy_result):
            if average_pairwise_distance(copy_result) <= average_pairwise_distance(T):
                T = copy_result
    """

    #print(len(G.__getitem__(0))) #number of neighbors in G
    #len(G.__getitem__(0)) - len(T.__getitem__(0))
    #print(G.__getitem__(1)) #get list of neighbors

    return T


def prim_mst_edges(G, start):
    """Iterate over edges of Prim's algorithm min/max spanning tree.
    Parameters
    ----------
    G : NetworkX Graph
        The graph holding the tree of interest.
    start : int
        The starting vertex to run Prim's on.
    """
    push = heappush
    pop = heappop
    nodes = set(G)
    c = count()

    while nodes:
        if (len(nodes) == G.number_of_nodes()): #let starting vertex be max_v
            u = start
            nodes.discard(u)
        else:
            u = nodes.pop()
        frontier = []
        visited = {u}

        for v, d in G.adj[u].items(): #all neighbor vertices v, d is weight of (u,v)
            """#heuristic stuff
            #sum of v's incident edges (v, w):
            sum_incident_edges = sum([d2.get("weight") for w, d2 in G.adj[v].items()])
            deg_v = len(G.__getitem__(v))
            score = 1
            print("score of vertex", v, score)
            """
            wt = d.get("weight") #edge weight
            push(frontier, (wt, next(c), u, v, d))

        while frontier:
            """ #heuristic stuff
            vertex_v = [v for w, _, u, v, d in frontier]

            #neighborhood of v:
            neighborhood_v = [set(G.__getitem__(v)) for w, _, u, v, d in frontier]
            dictionary = dict(zip(vertex_v, neighborhood_v))
            print(dictionary)

            #non-visited neighbors of v:
            print(visited.intersection(set(G.__getitem__(v))))
            #len(G.__getitem__(v)) - len(visited.intersection(set(G.__getitem__(v))))
            """

            W, _, u, v, d = pop(frontier)
            if v in visited or v not in nodes:
                continue
            yield u, v, d
            # update frontier
            visited.add(v)
            nodes.discard(v)
            for w, d2 in G.adj[v].items():
                if w in visited:
                    continue
                new_weight = d2.get("weight")
                push(frontier, (new_weight, next(c), v, w, d2))

if __name__ == '__main__':
    assert len(sys.argv) == 2
    total_pairwise_distance = 0
    #to run on all inputs: python3 solver.py all_inputs
    if sys.argv[1] == "all_inputs":
        
        for i in range(1,304):
            path = 'inputs/small-'+str(i)+'.in'
            G = read_input_file(path)
            T = solve(G)
            assert is_valid_network(G, T)
            total_pairwise_distance += average_pairwise_distance(T)
            print(path + " Average Pairwise Distance: {}".format(average_pairwise_distance(T)))
            path_string = re.split('[/.]', path)
            write_output_file(T, 'outputs/'+path_string[1]+'.out')
        print(" Total Average Small Pairwise Distance: {}".format(total_pairwise_distance/303))
        
        """
        for i in range(1,304):
            path = 'inputs/medium-'+str(i)+'.in'
            G = read_input_file(path)
            T = solve(G)
            assert is_valid_network(G, T)
            total_pairwise_distance += average_pairwise_distance(T)
            print(path + " Average Pairwise Distance: {}".format(average_pairwise_distance(T)))
            path_string = re.split('[/.]', path)
            write_output_file(T, 'outputs/'+path_string[1]+'.out')
        print(" Total Average Medium Pairwise Distance: {}".format(total_pairwise_distance/303))
        """
        """
        for i in range(1,401):
            path = 'inputs/large-'+str(i)+'.in'
            G = read_input_file(path)
            T = solve(G)
            assert is_valid_network(G, T)
            total_pairwise_distance += average_pairwise_distance(T)
            print(path + " Average Pairwise Distance: {}".format(average_pairwise_distance(T)))
            path_string = re.split('[/.]', path)
            write_output_file(T, 'outputs/'+path_string[1]+'.out')
        print(" Total Average Large Pairwise Distance: {}".format(total_pairwise_distance/400))
        """
        print(" Total Average Pairwise Distance: {}".format(total_pairwise_distance/1006))
        
    else:
        path = sys.argv[1]
        G = read_input_file(path)
        T = solve(G)
        assert is_valid_network(G, T)
        print(path + " Average pairwise distance: {}".format(average_pairwise_distance(T)))
        path_string = re.split('[/.]', path)
        write_output_file(T, 'outputs/'+path_string[1]+'.out')


