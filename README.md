# COVID19-Drug-Repurposing
A course project to re-purpose FDA approved drugs for COVID-19 using RGCN. The original RGCN code for link prediction can be found on [DGL github repo](https://github.com/dmlc/dgl/blob/master/examples/pytorch/rgcn/link_predict.py).

<img width="468" alt="graph" src="https://user-images.githubusercontent.com/7066351/81736743-0c77b180-9465-11ea-922e-000144ef90c8.png">

This project contains 3 main steps: data preprocessing, model training, and model evaluation.

## Data preprocessing

The datasets were obtained from:

- [DrugBank](https://www.drugbank.ca/), which contains drug-gene interactions
- [DisGeNET](https://www.disgenet.org/), which contains disease-gene interactions
- [Drug Central](http://drugcentral.org/), which contains disease-drug interactions
- [StringDB](https://string-db.org/), which contains gene-gene interactions

The final graph contains:

<img width="469" alt="final_graph" src="https://user-images.githubusercontent.com/7066351/81736854-39c45f80-9465-11ea-8ff8-85b12890d439.png">

which is stored in `data/wse.npy` in forms of triplets.

The graph is further pruned using `prune_graph` function from `model/prune.py`. The current pruned model can be found in `model/pruned_graph_1.npy` with details in `pruned_graph_info.txt`.

## Model training

The training code can be found in `run_model.py` with the following parameters:

```
model_param = {
    'h_dim'    : 80, # output feature size
    'dropout'  : 0.2,
    'num_bases': 30,
    'num_hidden_layers': 4,
    'use_cuda' : True,
    'reg_param': 0.01
}

# for sampling in each epoch
sample_graph_param = {
    'sample_size'  : 50000, # edges to sample
    'split_size'   : 0.5,
    'negative_rate': 1,
}

max_epoch = 1800
epoch_mult_eval = 20 # calculate MRR every 20 epochs
```
It generates a list of loss values and MRRs in csv files once the training is completed.

## Model evaluation

The code for evaluation can be found in `model_eval.ipynb` where it calculates the DistMult scores for edges between drugs and COVID-19 node and compare with the retrieved potential drugs obtained from Gordon et. al. (the list can be found in `data/clean/covid-drugs`).

The top 50 drugs are shown below.

<img width="1380" alt="results" src="https://user-images.githubusercontent.com/7066351/81737676-7775b800-9466-11ea-8fb9-4225b555148f.png">
