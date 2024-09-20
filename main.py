import argparse
from models import GCN_Net, ChebNet, APPNP_Net
from utils import load_planetoid_data, perturb_edges, calc_density, load_actor, random_planetoid_splits, load_squirrel, load_chameleon
from torch_geometric.utils import dropout_edge, dense_to_sparse
from enhanced_network import enhance_adjacency
import math
import torch.nn.functional as F
import torch

def run_experiment(args, dataset, data, Net):

    def train(model, optimizer, data, dprate):
        model.train()
        optimizer.zero_grad()
        out = model(data)[data.train_mask]
        nll = F.nll_loss(out, data.y[data.train_mask])
        loss = nll
        loss.backward()

        optimizer.step()
        del out

    def test(model, data):
        model.eval()
        logits, accs, losses, preds = model(data), [], [], []
        for _, mask in data('train_mask', 'val_mask', 'test_mask'):
            pred = logits[mask].max(1)[1]
            acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

            loss = F.nll_loss(model(data)[mask], data.y[mask])

            preds.append(pred.detach().cpu())
            accs.append(acc)
            losses.append(loss.detach().cpu())
        return accs, preds, losses
    
    network = Net(dataset=dataset, args=args)

    permute_masks = random_planetoid_splits

    #using default values
    train_rate = 0.025
    val_rate = 0.025
    percls_trn = int(round(train_rate*len(data.y)/dataset.num_classes))
    val_lb = int(round(val_rate*len(data.y)))
    TrueLBrate = (percls_trn*dataset.num_classes+val_lb)/len(data.y)

    data = permute_masks(data, dataset.num_classes, percls_trn, val_lb)

    model, data = network.to(device), data.to(device)

    if args.method in ['APPNP', 'GPRGNN','GARNOLDI','ARNOLDI']:
        optimizer = torch.optim.Adam([{
            'params': model.lin1.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.lin2.parameters(),
            'weight_decay': args.weight_decay, 'lr': args.lr
        },
            {
            'params': model.prop1.parameters(),
            'weight_decay': 0.0, 'lr': args.lr
        }
        ],
            lr=args.lr)
    else:
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.lr,
                                     weight_decay=args.weight_decay)
        
    best_val_acc = test_acc = 0
    best_val_loss = float('inf')
    val_loss_history = []
    val_acc_history = []

    if args.mode == 'verbose':
        print('DIAG: starting training')

    for epoch in range(args.epochs):
        train(model, optimizer, data, args.dprate)

        if (args.mode == 'verbose') & (epoch % 50 == 0):
            print(f'DIAG: starting epoch {epoch}')

        [train_acc, val_acc, tmp_test_acc], preds, [
            train_loss, val_loss, tmp_test_loss] = test(model, data)

        if val_loss < best_val_loss:
            best_val_acc = val_acc
            best_val_loss = val_loss
            test_acc = tmp_test_acc
            Alpha = args.alpha
            Gamma_0 = Alpha

        if epoch >= 0:
            val_loss_history.append(val_loss)
            val_acc_history.append(val_acc)
            if args.early_stopping > 0 and epoch > args.early_stopping:
                tmp = torch.tensor(
                    val_loss_history[-(args.early_stopping + 1):-1])
                if val_loss > tmp.mean().item():
                    break
    return test_acc, best_val_acc, Gamma_0

def perturbation_experiment(args, dataset, data, Net, p, enh_ratios):
    edge_index, _ = dropout_edge(data.edge_index, p, force_undirected=True)
    perturbed_data = data
    perturbed_data.edge_index = edge_index
    new_denisty = calc_density(data=perturbed_data)

    if args.mode == 'verbose':
        print('DIAG: starting perturb tests')
        print(f'perturbed density: {new_denisty}')

    p_test_acc, p_best_val_acc, p_Gamma_0 = run_experiment(args=args, dataset=dataset, data=perturbed_data, Net=Net)

    if args.mode == 'verbose':
        print('DIAG: baseline ran')

    e_test_results = []
    e_val_results = []

    for er in enh_ratios:
        if args.mode == 'verbose':
            print('DIAG: calculating enhanced matrix')
        (edge_index, edge_attr) = dense_to_sparse(enhance_adjacency(perturbed_data,new_denisty*er,args.k))
        enhanced_data = data
        enhanced_data.edge_index = edge_index
        if args.mode == 'verbose':
            print('DIAG: enh matrix calculated. starting experiment')
            enhanced_density = calc_density(data=enhanced_data)
            print(f'enhanced density: {enhanced_density}')
        e_test_acc, e_best_val_acc, e_Gamma_0 = run_experiment(args=args, dataset=dataset, data=enhanced_data, Net=Net)
        e_test_results.append(e_test_acc)
        e_val_results.append(e_best_val_acc)

    return p_test_acc, p_best_val_acc, e_test_results, e_val_results

if __name__ == '__main__':
    print("======  START OF TESTS  ======")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 0) ARGUMENTS MANAGEMENT
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, choices=['GCN', 'ChebNet', 'APPNP'], default='GCN')
    parser.add_argument('--dataset', type=str, choices=['cora', 'citeseer', 'pubmed', 'actor', 'squirrel','chameleon'], default='cora')
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=200)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--dprate', type=float, default=0.5)
    parser.add_argument('--k', type=str, choices=['single','double','triple'], default='double')
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--mode', type=str, choices=['verbose','normal'], default='normal')

    args = parser.parse_args()
    print(f'data: {args.dataset} | method: {args.method}')


    # 1.1) SETTING UP DATASET
    dataset_name = args.dataset

    if dataset_name in ['cora', 'citeseer', 'pubmed']:
        # load cora
        dataset, data = load_planetoid_data(dataset_name)
    elif dataset_name == 'actor': 
        dataset, data = load_actor()
    elif dataset_name == 'squirrel':
        dataset, data = load_squirrel()
    elif dataset_name == 'chameleon':
        dataset, data = load_chameleon()
    else: 
        # do sth else
        print("error while loading dataset")

    # 1.2) perform edge perturbations
        #probs, perturbs = perturb_edges(data=data)

    
    
    original_density = calc_density(data=data)
    print(f'original density: {original_density}')

    

    # SETTING UP MODEL
    method = args.method

    if method == 'APPNP':
        Net = APPNP_Net
    elif method == 'ChebNet':
        Net = ChebNet
    else:
        Net = GCN_Net

    
    #test_acc, best_val_acc, Gamma_0 = run_experiment(args=args, dataset=dataset, data=data, Net=Net)

    #print(f'original: test_acc: {test_acc}, best_val_acc: {best_val_acc}, gamma: {Gamma_0}')


    #perturb_probs = [0.1, 0.2, 0.4, 0.8]
    perturb_probs = [0]
    #enhance_ratio = [1.1, 1.2, 1.4, 1.8]
    enhance_ratio = [1.1, 1.5, 1.9]
 

    for p in perturb_probs:
        p_test_acc, p_best_val_acc, e_test_results, e_val_results = perturbation_experiment(args=args, dataset=dataset, data=data, Net=Net, p=p, enh_ratios=enhance_ratio)

        print(f'# for p = {p} ')
        
        print(f'   perturbed: test_acc= {p_test_acc} | best_val_acc= {p_best_val_acc}')

        for i in range(len(enhance_ratio)):
            e_test_acc = e_test_results[i]
            e_best_val_acc = e_val_results[i]
            er = enhance_ratio[i]
            print(f'   enhanced: test_acc= {e_test_acc} | best_val_acc= {e_best_val_acc} | ratio= {er}')


        # run on pertrubed


