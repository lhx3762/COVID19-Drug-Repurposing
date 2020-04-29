import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv

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
