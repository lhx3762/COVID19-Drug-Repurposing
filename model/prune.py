import numpy as np
import networkx as nx
from collections import Counter
import matplotlib.pyplot as plt
import json

covid_gene_dict = json.load(open('../data/clean/covid-gene-num-dict.wse.json'))
covid_gene_num = covid_gene_dict.values()

def prune_graph(init_graph, degree_thresh=100):
    '''Prune graph while ensuring COVID-19 genes are retained
    Arguments:
    - init_graph: a numpy list of triplets
    - degree_thresh: the degree threshold

    Returns a pruned graph as a numpy list of triplets
    '''
    G = nx.Graph()
    nx_edges = [(node1, node2) for (node1, node2) in zip(init_graph[:,0], init_graph[:,2])]
    G.add_edges_from(nx_edges)

    print('Pruning given degree threshold of %d...' % (degree_thresh))
    print('Initial # edges:', init_graph.shape[0])

    remove_nodes = [node for node, degree in dict(G.degree()).items() if degree < degree_thresh and node not in covid_gene_num]
    print('Removing %d nodes' % len(remove_nodes))

    final_graph = init_graph[~((np.isin(init_graph[:,0], remove_nodes)) | (np.isin(init_graph[:,2], remove_nodes))),:]
    print('The final # edges:', final_graph.shape[0])
    return(final_graph)
