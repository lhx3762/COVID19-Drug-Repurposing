import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl import heterograph
import dgl.function as fn
import dgl.utils as dgl_utils
from functools import partial
from dgl.nn.pytorch import RelGraphConv
from dgl.contrib.data import load_data

import numpy as np

from sklearn.model_selection import train_test_split

import time
import utils
from base import BaseRGCN

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        # create an embedding value [0,1] for each node of the graph
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())

class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        # build a number of hidden layer according to the parameter
        # add a relu activation function except for the last layer
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, 'basis',
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)

class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        """
        Arguments:
        - in_dim (int) -- input feature size
        - h_dim  (int) -- output feature size
        - num_rels (int) -- # relations
        - num_bases (int) -- # bases
        - num_hidden_layers (int) -- # hidden layers
        - dropout (float) -- [0,1] dropout rate
        - use_cuda (bool)
        - reg_param (float) -- regularization parameter
        """
        super(LinkPredict, self).__init__()
        # build RGCN layer
        # 2 x num_rels as both directions are considered
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        # define regularization
        self.reg_param = reg_param
        # define relations and normalize them
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # apply DistMult for scoring
        # embedding contains the embedding values of the node after the
        #   propagation within the RGCN Block layer
        # triplets contains all triples resulting from the negative sampling process
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)

        # The score is computed with the value-by-value multiplication of
        #   the embedding values of data produced by the negative sampling process
        #   and sum them using the vertical dimension

        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

#---------------------------

# load the graph
graph = np.load('../data/clean/graph.npy')
num_nodes = len(list(set(np.unique(graph[:,0])).union(set(np.unique(graph[:,2])))))
num_rels = np.unique(graph[:,1]).shape[0]
num_edges = graph.shape[0]

# split train, val, test
train_val, test_data = train_test_split(graph, test_size=0.1, random_state=0)
train_data, val_data = train_test_split(train_val, test_size=0.2, random_state=0)

val_data = torch.LongTensor(val_data)
test_data = torch.LongTensor(test_data)

# build a test graph
test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
test_deg = test_graph.in_degrees(
                range(test_graph.number_of_nodes())).float().view(-1, 1)
test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
test_rel = torch.from_numpy(test_rel)
test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

''' Params
'''
# model params
model_param = {
    'in_dim'   : num_nodes,
    'h_dim'    : 100, # output feature size
    'num_rels' : num_rels,
    'dropout'  : 0.2,
    'use_cuda' : True,
    'reg_param': 0.01
}
use_cuda = model_param['use_cuda']

sample_graph_param = {
    'sample_size'  : 50000, # edges to sample
    'split_size'   : 0.5,
    'negative_rate': 10,
}

if use_cuda: torch.device('cuda')

''' Params - end
'''

# create the model
model = LinkPredict(in_dim   = model_param['in_dim'],
                    h_dim    = model_param['h_dim'],
                    num_rels = model_param['num_rels'],
                    dropout  = model_param['dropout'],
                    use_cuda = model_param['use_cuda'],
                    reg_param= model_param['reg_param'])

if use_cuda:
    model.cuda()

# build adj list and calculate degrees for sampling
adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

epoch = 0
max_epoch = 20
epoch_mult_eval = 5 # multiplication of n epochs to indicate when to evaluate
best_mrr = 0
forward_time = []
backward_time = []
eval_batch = 500
model_state_file = 'model_state_gpu.pth'

print("start training...")
while True:
    model.train()
    epoch += 1

    # Perform edge neighborhood sampling to generate training graph and data
    # The training stage is performed on a sample graph (not the entire graph)
    g, node_id, edge_type, node_norm, data, labels = \
        utils.generate_sampled_graph_and_labels(
            train_data,
            sample_graph_param['sample_size'],
            sample_graph_param['split_size'],
            num_rels,
            adj_list,
            degrees,
            sample_graph_param['negative_rate'])

    print("Finished edge sampling")

    # set node/edge feature
    node_id = torch.from_numpy(node_id).view(-1, 1).long()
    edge_type = torch.from_numpy(edge_type)
    edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
    data, labels = torch.from_numpy(data), torch.from_numpy(labels)
    deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
    if use_cuda:
        node_id, deg = node_id.cuda(), deg.cuda()
        edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
        data, labels = data.cuda(), labels.cuda()

    t0 = time.time()
    embed = model(g, node_id, edge_type, edge_norm)
    loss = model.get_loss(g, embed, data, labels)
    t1 = time.time()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # clip gradients
    optimizer.step()
    t2 = time.time()

    forward_time.append(t1 - t0)
    backward_time.append(t2 - t1)
    print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
              format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

    optimizer.zero_grad()

    # validation
    if epoch % epoch_mult_eval == 0:
        print('Run evaluation...')
        # perform validation on CPU because full graph is too large
        if use_cuda: model.cpu()
        model.eval()
        embed = model(test_graph, test_node_id, test_rel, test_norm)
        mrr = utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
                             val_data, test_data, hits=[1, 3, 10], eval_bz=eval_batch,
                             eval_p='filtered')
        # save best model
        if mrr < best_mrr:
            if epoch >= max_epoch:
                break
        else:
            best_mrr = mrr
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       model_state_file)
        if use_cuda:
            model.cuda()
