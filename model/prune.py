import numpy as np
import networkx as nx
from collections import Counter
import json
import pandas as pd

# obtain covid gene dictionary
covid_gene_dict = json.load(open('../data/clean/covid-gene-num-dict.wse.json'))
covid_gene_nums = covid_gene_dict.values()

covid_drug_dict = json.load(open('../data/clean/covid-drug-num-dict.wse.json'))
covid_drug_nums = covid_drug_dict.values()

# obtain covid-19 genes with drug targets
gene_drug_targets = pd.read_csv('../data/clean/covid-gene-drug-targets', header=None)
gene_drug_targets = list(set(gene_drug_targets[0].tolist()))
gene_drug_targets_num = []
for gene in gene_drug_targets:
    gene_drug_targets_num.append(covid_gene_dict[gene])

def get_neighbors(G, covid_neighbors, node, hop, max_hops=2):
    '''Obtain direct neighbors given node until the max_hops is reached
    '''
    if node in covid_neighbors or hop > max_hops: return
    covid_neighbors.append(node)
    for neighbor in dict(G[node]).keys():
        get_neighbors(G, covid_neighbors, neighbor, hop+1, max_hops)

def prune_graph(init_graph,
                disease_degree_thresh=200,
                drug_degree_thresh=200,
                gene_degree_thresh=500,
                covid_neighbor_hops=2):
    '''Prune graph while ensuring COVID-19 genes are retained:
    1. obtain covid-19 gene neighbors given # hops
    2. prune the graph by degree thresholds for each type, retaining the neighbors from (1)
    3. keep the largest graph (if multiple components are produced)
    4. add COVID-19 disease

    Arguments:
    - init_graph           : a numpy list of triplets
    - disease_degree_thresh: the degree threshold for disease nodes
    - drug_degree_thresh   : the degree threshold for drug nodes
    - gene_degree_thresh   : the degree threshold for gene nodes
    - covid_neighbor_hops  : # hops from covid genes

    Returns a pruned graph as a numpy list of triplets
    '''
    # create networkx graph
    G = nx.Graph()
    nx_edges = [(node1, node2) for (node1, node2) in zip(init_graph[:,0], init_graph[:,2])]
    G.add_edges_from(nx_edges)
    print('Initial # edges:', init_graph.shape[0])

    print('Pruning given degree threshold of %d (disease), %d (drug), and %d (gene)...' %
          (disease_degree_thresh, drug_degree_thresh, gene_degree_thresh))

    # 1. obtain covid-19 gene neighbors given # hops
    covid_neighbors = []
    covid_nodes = list(covid_gene_nums) + list(covid_drug_nums)
    for node_num in covid_nodes:
        get_neighbors(G, covid_neighbors, node_num, 0, covid_neighbor_hops)
    print('Given %d hops, %d neighboring nodes are obtained.' % (covid_neighbor_hops, len(covid_neighbors)))

    # 2. prune the graph by degree thresholds for each type
    drug_num = 8079 # 1st in edges
    disease_num = 11171 # 2nd in edges
    gene_num = 18643 # 3rd in edges
    remove_nodes = [node for node, degree in dict(G.degree()).items()\
                    # drugs - disease - gene
                    if ((node < drug_num and degree < drug_degree_thresh) or\
                        (node >= drug_num and node < (drug_num + disease_num) and degree < disease_degree_thresh) or\
                        (node >= (drug_num + disease_num) and degree < gene_degree_thresh)) and\
                        node not in covid_gene_nums and node not in covid_neighbors]
    print('Removing %d nodes...' % len(remove_nodes))

    final_graph = init_graph[~((np.isin(init_graph[:,0], remove_nodes)) | (np.isin(init_graph[:,2], remove_nodes))),:]
    G2 = nx.Graph()
    G2.add_edges_from([(node1, node2) for (node1, node2) in zip(final_graph[:,0], final_graph[:,2])])

    # 3. keep the largest graph
    included_nodes = []
    for component in sorted(nx.connected_components(G2), key=len, reverse=True):
        included_nodes = list(component)
        break
    final_graph = final_graph[np.isin(final_graph[:,0], included_nodes) & np.isin(final_graph[:,2], included_nodes),:]
    included_covid_genes = list(filter(lambda x: x in covid_gene_nums, included_nodes))
    included_covid_drugs = list(filter(lambda x: x in covid_drug_nums, included_nodes))
    # 4. add COVID-19 disease
    covid_disease_node = drug_num + disease_num + gene_num # the last node
    add_new_node = False
    if covid_disease_node not in included_nodes:
        add_new_node = True
        print('Adding COVID-19 node: %d' % covid_disease_node)
        covid_graph = np.zeros((len(included_covid_genes), 3), dtype=int)
        gene_disease_relation = 12
        for i, gene_num in enumerate(included_covid_genes):
            covid_graph[i] = [gene_num, gene_disease_relation, covid_disease_node]
        final_graph = np.concatenate((final_graph, covid_graph), axis=0)
        included_nodes = included_nodes + [covid_disease_node]
    new_disease_node = 1 if add_new_node else 0

    # print the details of the final graph
    print('The final graph contains:')
    print('- %d edges' % final_graph.shape[0])
    print('- %d nodes: %d diseases, %d genes, %d drugs' % (
            len(included_nodes),
            len(list(filter(lambda x: x >= drug_num and x < drug_num+disease_num, included_nodes)))+new_disease_node,
            len(list(filter(lambda x: x >= drug_num+disease_num, included_nodes))),
            len(list(filter(lambda x: x < drug_num, included_nodes)))))
    print('- %d covid-19-associated genes (out of 312)' % len(included_covid_genes))
    print('- %d covid-19 associated genes + drug targets (out of 62)' %
            len(list(filter(lambda x: x in gene_drug_targets_num, included_nodes))))
    print('- %d covid-19-associated drugs (out of 26)' % len(included_covid_drugs))
    return(final_graph)
