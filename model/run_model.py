import torch
import numpy as np

from sklearn.model_selection import train_test_split

import time
import utils
from model import LinkPredict

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
    return g.edata['norm']

#---------------------------
# Parameters
#---------------------------

model_param = {
    'h_dim'    : 100, # output feature size
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

max_epoch = 50
epoch_mult_eval = 50 # multiplication of n epochs to indicate when to evaluate

mrr_param = {
    'eval_batch': 500,
    'eval_p': 'filtered',
    'hits': [1, 10],
}

eval_batch = 500

#---------------------------
# Pipeline
#---------------------------

# load the graph
graph = np.load('../data/clean/graph.wse.npy')

num_nodes = len(list(set(np.unique(graph[:,0])).union(set(np.unique(graph[:,2])))))
num_rels = np.unique(graph[:,1]).shape[0]
num_edges = graph.shape[0]

print('There are %d nodes, %d rels, %d edges' % (num_nodes, num_rels, num_edges))

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

# create the model
model = LinkPredict(in_dim   = num_nodes,
                    h_dim    = model_param['h_dim'],
                    num_rels = num_rels,
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

best_mrr = -1
forward_time = []
backward_time = []
loss_by_epoch = []

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
                             val_data, test_data, hits=mrr_param['hits'], eval_bz=mrr_param['eval_batch'],
                             eval_p=mrr_param['eval_p'])
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

loss_df = pd.DataFrame({ 'epoch': range(1, len(loss_by_epoch)+1), 'loss': loss_by_epoch })
loss_df.to_csv('loss.csv', index=False)
