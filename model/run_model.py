import torch
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import time
import utils
from model import LinkPredict
from prune import prune_graph

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

# def get_ROC(scores_pos, scores_neg):
#     def sigmoid(x):
#         return x
#         return 1 / (1 + np.exp(-x))

#     preds = []
#     for score in scores_pos:
#         preds.append(sigmoid(score))
#     for score in scores_neg:
#         preds.append(sigmoid(score))
#     labels = np.hstack([np.ones(len(scores_pos)), np.zeros(len(scores_neg))])
#     roc_score = roc_auc_score(labels, preds)
#     return(roc_score)

#---------------------------
# Parameters
#---------------------------

model_name = 'model_pruned_graph_1'

model_param = {
    'h_dim'    : 50, # output feature size
    'dropout'  : 0.2,
    'num_bases': 30,
    'num_hidden_layers': 12,
    'use_cuda' : True,
    'reg_param': 0.01
}
use_cuda = model_param['use_cuda']

sample_graph_param = {
    'sample_size'  : 50000, # edges to sample
    'split_size'   : 0.5,
    'negative_rate': 8,
}

if use_cuda: torch.device('cuda')
torch.cuda.empty_cache()

max_epoch = 5000
epoch_mult_eval = 20 # multiplication of n epochs to indicate when to evaluate

mrr_param = {
    'eval_batch': 500,
    'eval_p': 'raw',
    'hits': [1, 10],
}

#---------------------------
# Pipeline
#---------------------------

# load and prune the graph
# graph = np.load('../data/clean/graph.wse.npy')
# graph = prune_graph(graph, disease_degree_thresh=700, drug_degree_thresh=100, gene_degree_thresh=1000, covid_neighbor_hops=1)
# np.save('pruned_graph.npy', graph)
graph = np.load('./pruned_graph_1.npy')

# map nodes
all_nodes = sorted(list(set(graph[:,0].tolist() + graph[:,2].tolist())))
node_map = {i: node for i, node in enumerate(all_nodes)}
node_map_rev = {node: i for i, node in enumerate(all_nodes)}

# map the nodes
mapped_graph = np.zeros(graph.shape, dtype=int)
for i, row in enumerate(graph):
    mapped_graph[i] = [node_map_rev[row[0]], row[1], node_map_rev[row[2]]]

diseases_nodes = list(set(mapped_graph[np.isin(mapped_graph[:,1], [0])][:,2]))
drugs_nodes = list(set(mapped_graph[np.isin(mapped_graph[:,1], range(5))][:,0]))

num_nodes = len(list(set(np.unique(mapped_graph[:,0])).union(set(np.unique(mapped_graph[:,2])))))
num_rels = np.unique(mapped_graph[:,1]).shape[0]
num_edges = mapped_graph.shape[0]

# divide drug-treat-disease associations into test, train, and validation
# initiate train data with non-treat edges (relation type: 0)
train_data = mapped_graph[mapped_graph[:,1] != 0]
full_treat_edges = mapped_graph[mapped_graph[:,1] == 0]
treat_train_data, val_data = train_test_split(full_treat_edges, test_size=0.2, random_state=0)
train_data = np.concatenate((train_data, treat_train_data), axis=0)

# add negative validation data
val_neg_data = []
for i in range(len(val_data)*7):
    while True:
        drug = np.random.choice(drugs_nodes)
        disease = np.random.choice(diseases_nodes)
        if len(full_treat_edges[np.isin(full_treat_edges[:,0], [drug]) &
                                np.isin(full_treat_edges[:,2], [disease])]) == 0:
            val_neg_data.append([drug, 0, disease])
            break
val_data = torch.LongTensor(val_data)
val_neg_data = torch.from_numpy(np.array(val_neg_data, dtype=np.int64))

# build a test graph
test_graph, test_rel, test_norm = utils.build_test_graph(
        num_nodes, num_rels, train_data)
test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
test_rel = torch.from_numpy(test_rel)
test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

# create the model
model = LinkPredict(in_dim   = num_nodes,
                    h_dim    = model_param['h_dim'],
                    num_rels = num_rels,
                    num_bases= model_param['num_bases'],
                    num_hidden_layers = model_param['num_hidden_layers'],
                    dropout  = model_param['dropout'],
                    use_cuda = model_param['use_cuda'],
                    reg_param= model_param['reg_param'])

if use_cuda:
    model.cuda()

# build adj list and calculate degrees for sampling
adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epoch = 0

best_mrr = -1
best_roc = -1
forward_time = []
backward_time = []
loss_by_epoch = []

model_state_file = model_name + '.model_state_gpu.pth'

print("start training...")
while True:
    model.train()
    epoch += 1
    if epoch > max_epoch: break

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
    if use_cuda:
        node_id = node_id.cuda()
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

    loss_by_epoch.append(loss.item())
    optimizer.zero_grad()

    # validation
    if epoch % epoch_mult_eval == 0:
        print('Run evaluation...')
        # perform validation on CPU because full graph is too large
        if use_cuda: model.cpu()
        model.eval()
        embed = model(test_graph, test_node_id, test_rel, test_norm)
        mrr = utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
                             [], val_data, hits=mrr_param['hits'], eval_bz=mrr_param['eval_batch'],
                             eval_p=mrr_param['eval_p'])
        # save best model
        if mrr > best_mrr:
            best_mrr = mrr
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                        model_state_file)
        if use_cuda:
            model.cuda()
        
loss_df = pd.DataFrame({ 'epoch': range(1, len(loss_by_epoch)+1), 'loss': loss_by_epoch })
loss_df.to_csv(model_name + '.loss.csv', index=False)
