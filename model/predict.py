import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import heterograph
import dgl.function as fn
import dgl.utils as dgl_utils
from functools import partial
from dgl.nn.pytorch import RelGraphConv

import numpy as np
import pandas as pd
import pygraphviz as pgv

from collections import Counter, OrderedDict
import matplotlib.pyplot as plt
import time
import utils
from base import BaseRGCN

#######################################################################
#
# To get DB ID of covid19 drugs with top 100 edge scores, call this function as:
# drug_db = predict_drug(graph, model, embed, topk=100)
#
#######################################################################

def predict_drug(graph, model, embed, covid19_num = 37893, topk=100):

    # drug-number mapping (kathleen's code)
    drugs = pd.read_csv('../data/clean/drugs.wse.nodes', header=None)
    num_drugs = drugs.shape[0]
    covid_drugs = pd.read_csv('../data/clean/covid-drugs', header=None)
    drug_num_dict_rev = {num: drug for (drug, num) in zip(drugs[0], range(num_drugs))}

    graph1 = graph[graph[:,0]<8079]
    graph2 = graph[graph[:,2]<8079]
    drug = list(set(np.unique(graph1[:,0])).union(set(np.unique(graph2[:,2]))))

    # create test edges for covid19
    covid_test_edges = []
    for i in range(len(drug)):
        edge =[drug[i],0,covid19_num]
        covid_test_edges.append(edge)
    covid_test_edges = np.array(covid_test_edges)

    # calcuate scores for test edges
    test_score = model.calc_score(embed, covid_test_edges)
    # sort sscores
    idx = torch.flip(torch.argsort(test_score), dims=[0])
    scores = test_score[idx].detach().cpu().numpy()
    # get drug index with topk scores
    drug_topk = idx[:topk]
    drug_topk = drug_topk.detach().cpu().numpy()
    # map number back to DB ID
    drug_db = []
    for i in drug_topk:
        drug_db.append(drug_num_dict_rev[i])
    
    # get intersection of drug_topk and covid_drugs
    inter = list(set(covid_drugs[0]) & set(drug_db))
    print('intersection of drug_topk and covid_drugs:', inter)

    return drug_db
  
