import networkx as nx
from parse import read_input_file, write_output_file
from utils import is_valid_network, average_pairwise_distance
import sys

import re
import matplotlib.pyplot as plt
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

    """
    plt.figure(figsize=(8,8))
    pos=nx.spring_layout(G)
    nx.draw_networkx(G, pos)
    labels = nx.get_edge_attributes(G,"weight")
    nx.draw_networkx_edge_labels(G,pos,edge_labels=labels)
    nx.draw_networkx_edges(G,pos, edgelist=[e for e in T.edges()], width=5,alpha=0.5,edge_color='r')
    plt.show()
    """

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
    available_edges = len(G.nodes) - 1
    dominated = set() #set of dominated vertices
    num_edges_mst = 0

    while num_edges_mst < len(G.nodes):
        i = 0
        u,v,d = edges_copy[0]
        while (u in dominated and v in dominated and i + 1 < len(edges_copy) and subtrees[u] == subtrees[v]):
            i += 1
            u,v,d = edges_copy[i]
        if subtrees[u] != subtrees[v]:
            dominated.add(u)
            dominated.add(v)
            new_v_reached = 0
            u_subtree = []
            v_subtree = []

            #for loop to find the number of new vertices reached:
            for x,y,w in edges_copy:
                if x in set(G.__getitem__(u)) or y in set(G.__getitem__(v)): #neighbors of u and v
                    if x not in dominated:
                        new_v_reached += 1
                    if y not in dominated:
                        new_v_reached += 1

            #all parts of current mst
            subtrees.union(u, v)
            curr_mst_vertices = [list(s) for s in subtrees.to_sets() if u in s]
            curr_mst_vertices = curr_mst_vertices[0]
            current_tree = nx.Graph()
            for v1,v2,v3 in G.edges.data():
                if v1 in curr_mst_vertices and v2 in curr_mst_vertices:
                    edges_ = edges[edges_no_weights.index((v1, v2))]
                    current_tree.add_edge(v1, v2, weight=v3['weight'])
            before = average_pairwise_distance(current_tree)

            #add one edge to find increase in cost
            for v1,v2,v3 in edges_copy:
                #print(v1,v2,v3)
                new_tree = current_tree.copy()
                #edge (u, X) (v, X) (X, u) (X, v)
                if (v1 == u and v2 != v) or (v1 == v and v2 != v) or (v2 == u and v1 != v) or (v2 == v and v1 != u):
                    new_tree.add_edge(v1, v2, weight=v3['weight'])
                    after = average_pairwise_distance(new_tree)
                    edge_update = edges_copy[edges_no_weights.index((v1,v2))]
                    edge_update_list = list(edge_update[0:2])
                    if after - before > 0 and new_v_reached!=0:
                        edge_update_list.append({'weight':((after-before)/new_v_reached)})
                        edges_copy[edges_no_weights.index((v1,v2))] = edge_update_list
            #update edges_copy
            edges_copy = sorted(edges_copy, key=lambda x:x[2]['weight'])
            #available_edges -= 1
            num_edges_mst += 1
            yield (u, v, edges[edges_no_weights.index((u,v))][2])
            if num_edges_mst == len(G.nodes)-1:
                break
        #available_edges -= 1

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
            wt = d.get("weight") #edge weight
            push(frontier, (wt, next(c), u, v, d))

        while frontier:
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
        
        for i in range(266,304):
            path = 'inputs/small-'+str(i)+'.in'
            G = read_input_file(path)
            T = solve(G)
            assert is_valid_network(G, T)
            total_pairwise_distance += average_pairwise_distance(T)
            print(path + " Average Pairwise Distance: {}".format(average_pairwise_distance(T)))
            path_string = re.split('[/.]', path)
            write_output_file(T, 'outputs/'+path_string[1]+'.out')
        print(" Total Average Small Pairwise Distance: {}".format(total_pairwise_distance/303))

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


