import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys

import re
from networkx.algorithms import tree
from networkx.utils import UnionFind
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
    all_max_deg_vs = [v for v in v_descending_degree if G.degree(v)==G.degree(v_descending_degree[0])]

    curr_max_deg_v = v_descending_degree[0]
    curr_min_sum = sum([v3['weight'] for v1,v2,v3 in G.edges.data() if v1==curr_max_deg_v or v2==curr_max_deg_v])
    for v in all_max_deg_vs:
        sum_incident_edges = sum([v3['weight'] for v1,v2,v3 in G.edges.data() if v1==v or v2==v])
        if (sum_incident_edges <= curr_min_sum):
            curr_max_deg_v = v
            curr_min_sum = sum_incident_edges

    max_v = curr_max_deg_v

    if len(G.__getitem__(max_v)) == G.number_of_nodes()-1:
        T = nx.Graph()
        T.add_node(max_v)
        return T

    mst_edges = kruskal_mst_edges(G)
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

    return T


def kruskal_mst_edges(G, weight='weight', data=True):
    """Generate edges in a minimum spanning forest of an undirected
    weighted graph.
    Parameters
    ----------
    G : NetworkX Graph
    weight : string
       Edge data key to use for weight (default 'weight').
    data : bool, optional
       If True yield the edge data along with the edge.
    Returns
    -------
    edges : iterator
       A generator that produces edges in the minimum spanning tree.
       The edges are three-tuples (u,v,w) where w is the weight.
    """
    subtrees = UnionFind()
    edges = sorted(G.edges(data=True), key=lambda t: t[2].get("weight")) #sorted by edge weights first
    edges_no_weights = [e[0:2] for e in edges] #same order as edges
    edges_copy = edges.copy()
    available_edges = len(edges)
    print(available_edges)

    print(edges.index((3, 23, {'weight': 5.037})))
    dominated = set() #set of dominated vertices

    while (available_edges > 0):
        u,v,d = edges_copy[0]
        if subtrees[u] != subtrees[v]:
            dominated.add(u)
            dominated.add(v)
            #size_dominated = len(dominated) #currently dominated
            new_v_reached = 0
            for x,y,w in edges_copy:
                if x in set(G.__getitem__(u)) or y in set(G.__getitem__(v)):
                    if x not in dominated:
                        new_v_reached += 1
                    if y not in dominated:
                        new_v_reached += 1
                print(list(subtrees.to_sets()))
                u_subtree = [s for s in list(subtrees.to_sets()) if u in s]
                v_subtree = [s for s in list(subtrees.to_sets()) if v in s]

                print("u subtree",u_subtree)
                print("v subtree",v_subtree)
                new_tree = nx.Graph()
                for v1,v2,v3 in G.edges.data():
                    if v1 in u_subtree and v2 in v_subtree:
                        new_tree.add_edge(v1, v2, weight=v3['weight'])
                    elif v2 in v_subtree and v1 in u_subtree:
                        new_tree.add_edge(v2, v1, weight=v3['weight'])
                    #print(new_tree.edges(data=True))
                #before = average_pairwise_distance(new_tree)
            yield (u, v, d)
            available_edges -= 1
            subtrees.union(u, v)


    #print(visited.intersection(set(G.__getitem__(v))))
    #len(G.__getitem__(v)) - len(visited.intersection(set(G.__getitem__(v))))
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

