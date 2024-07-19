from torch_geometric.datasets import Planetoid, Actor, WikipediaNetwork
import os.path as osp
import torch_geometric.transforms as T
from torch_geometric.utils import dropout_edge
import torch

def load_planetoid_data(name):
    root_path = '../'
    path = osp.join(root_path, 'data', name)
    dataset = Planetoid(path, name, transform=T.NormalizeFeatures())

    return dataset, dataset[0]

def load_actor():
    root_path = '../'
    path = osp.join(root_path, 'data', 'actor')
    dataset = Actor(path, transform=T.NormalizeFeatures())

    return dataset, dataset[0]

def load_squirrel():
    preProcDs = WikipediaNetwork(
            root='../data/', name='squirrel', geom_gcn_preprocess=False, transform=T.NormalizeFeatures())
    dataset = WikipediaNetwork(
            root='../data/', name='squirrel', geom_gcn_preprocess=True, transform=T.NormalizeFeatures())
    data = dataset[0]
    data.edge_index = preProcDs[0].edge_index
    return dataset, data

def perturb_edges(data, probs=[0, 0.1, 0.2, 0.4]): #TODO
    perturbations = []
    for p in probs:
        edge_index, _ = dropout_edge(data.edge_index, p, force_undirected=True)
        perturbations.append(edge_index)

        
    return probs, perturbations

def calc_density(data):
    n = float(data.num_nodes)
    if data.is_directed():
        e = data.num_edges
        possible = n*n
    else:
        e = data.num_edges/2
        possible = n*(n-1)*0.5
    density = float(e) / possible
    return density

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask

def random_planetoid_splits(data, num_classes, percls_trn=20, val_lb=500, Flag=0):
    # Set new random planetoid splits:
    # * round(train_rate*len(data)/num_classes) * num_classes labels for training
    # * val_rate*len(data) labels for validation
    # * rest labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:percls_trn] for i in indices], dim=0)

    if Flag == 0:
        rest_index = torch.cat([i[percls_trn:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(rest_index[:val_lb], size=data.num_nodes)
        data.test_mask = index_to_mask(
            rest_index[val_lb:], size=data.num_nodes)
    else:
        val_index = torch.cat([i[percls_trn:percls_trn+val_lb]
                               for i in indices], dim=0)
        rest_index = torch.cat([i[percls_trn+val_lb:] for i in indices], dim=0)
        rest_index = rest_index[torch.randperm(rest_index.size(0))]

        data.train_mask = index_to_mask(train_index, size=data.num_nodes)
        data.val_mask = index_to_mask(val_index, size=data.num_nodes)
        data.test_mask = index_to_mask(rest_index, size=data.num_nodes)
    return data