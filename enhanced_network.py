import torch 
from torch_geometric.utils import to_dense_adj, dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import math
import numpy as np



def calculate_posteriors(data):
    set_of_classes = torch.unique(data.y) #TODO Convert to multilabel 
    A = to_dense_adj(edge_index=data.edge_index, max_num_nodes=data.num_nodes)
    #experimental
    A_2 = torch.matmul(A,A)
    #A_2 = torch.gt(A,0)
    A_3 = torch.matmul(A,A_2)
    #A_3 = torch.gt(A_3,0)
    A = A + A_2 + A_3
    A = torch.gt(A,0)
    #end of experimental

    posteriors = torch.zeros(data.num_nodes, len(set_of_classes)) #intialize
    
    #use only training labels (set others to -1 to ignore)
    labels = torch.ones_like(data.y)
    labels = torch.mul(labels, -1)
    labels[data.train_mask] = data.y[data.train_mask]

    for c in set_of_classes:
        V_c = (labels == c).nonzero().squeeze()   
        M_c = A[:,V_c].squeeze().t() #line 3 of alg1
        #M_c = torch.index_select(A,1,V_c)
        gamma_c = torch.sum(M_c,1).reshape(1,-1) #line 4
        H_c = torch.ge(gamma_c.t(), gamma_c) #line 5
        all = torch.sum(H_c, 1) #line 7
        temp = torch.index_select(H_c,1,V_c)
        in_class = torch.sum(temp,1) #line 6
        posterior_c = torch.div(in_class, all) #line 8
        posterior_c[V_c] = 1 #line 9
        posteriors[:,c] = posterior_c #line 10

    posteriors[torch.isnan(posteriors)] = 0 #might be a better way to ensure this doesnt happen
    posteriors[torch.isinf(posteriors)] = 0 # same
    return posteriors

def _unravel(index, shape): # takes in flat index and converts to n-dim tensor indices
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))



def enhance_adjacency(data, delta):
    #TODO: use only training data
    A = to_dense_adj(edge_index=data.edge_index, max_num_nodes=data.num_nodes).squeeze()
    e = torch.count_nonzero(A)/2 #assumes undirected
    n = data.num_nodes
    target = math.ceil(n*(n-1)*(0.5)*delta) 
    e_hat = target - e
    e_hat = int(e_hat.item())
    if e_hat < 1:
        raise ValueError("invalid delta")
    Q = torch.zeros(n,n)
    set_of_classes = torch.unique(data.y)
    posteriors = calculate_posteriors(data)
    for c in set_of_classes:
        post = torch.index_select(posteriors,1,c)
        Q_c = torch.matmul(post, post.t())
  
        Q = torch.max(Q,Q_c)


    Q = Q - A  
    Q = torch.triu(Q,diagonal=1)
    #Q = Q.squeeze()

    #ind = np.diag_indices(Q.shape[0]) #Set diagonal entries to 0
    #Q[ind[0], ind[1]] = torch.zeros(Q.shape[0])
    
    idx  = torch.tensor([_unravel(k, Q.size()) for k in torch.topk(Q.flatten(), e_hat).indices]) #might be a better way to do it. 
    #v, i = torch.topk(Q.flatten(), e_hat, sorted=True)
    #idx = np.array(np.unravel_index(i.numpy(), Q.shape)).T


    result = A
    row = torch.index_select(idx,1,torch.tensor([0]))
    col = torch.index_select(idx,1,torch.tensor([1]))
    result[row, col] = 1 #TODO maybe average of nonzero values in case of weighted
    temp = torch.triu(result, diagonal=1)
    result = temp + temp.t()
    
    return result


def test():
    print("start of tests:")
    print("========================")
    print("toy example: n=6, y=[0-1]")
    edge_index = torch.tensor([[0,2],
                              [2,0],
                              [1,3],
                              [3,1],
                              [2,3],
                              [3,2],
                              [2,4],
                              [4,2],
                              [3,4],
                              [4,3],
                              [4,5],
                              [5,4]],
                              dtype=torch.long)
    x = torch.rand(6,1)
    y = torch.tensor([1,1,0,0,0,1])
    data = Data(x=x, edge_index=edge_index.t().contiguous(), y=y)
    data.validate(raise_on_error=True)
    print(data)
    post = calculate_posteriors(data)
    print(post)
    new_adj = enhance_adjacency(data, 0.5)
    print("enhanced adj:")
    print(new_adj)
    (edge_index, edge_attr) = dense_to_sparse(new_adj)
    data.edge_index = edge_index

    data.validate(raise_on_error=True)
    print(data.num_edges)
    print(data.num_node_types)
    print(torch.unique(data.y))

    density = data.num_edges/(data.num_nodes * data.num_nodes)
    print(f'density: {density}')


    print("===========================")
    print("Cora test")
 
    dataset = Planetoid(root='data/Planetoid', name='Cora', transform=NormalizeFeatures())
    data = dataset[0]
    print(data)
    post = calculate_posteriors(data)
    density = data.num_edges/(data.num_nodes * data.num_nodes)
    print(f'density: {density}')
    print(post)
    print(f'post: {len(torch.unique(post))}')

    new_adj = enhance_adjacency(data, 0.002)
    print("enhanced adj:")
    print(new_adj)
    (edge_index, edge_attr) = dense_to_sparse(new_adj)
    data.edge_index = edge_index
    data.validate(raise_on_error=True)
    print(data.num_edges)
    print(data.num_node_types)
    print(torch.unique(data.y))
    density = data.num_edges/(data.num_nodes * data.num_nodes)
    print(f'density: {density}')



if __name__ == "__main__":
    test()