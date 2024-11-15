"""
Code implementation is largely based on Hwang et al., CVPR 2024.
Refer to https://github.com/y0ngjaenius/CVPR2024_FLOWERFormer.
"""

import sys
import json
import yaml
import warnings
import argparse

import torch
import copy
from tqdm import trange
from torch.utils.data import Dataset
import pickle

from utils import (
    cosine_with_warmup_scheduler,
    seed_everything,
    Option,
    get_data,
    get_data_2cell,
    get_perms)


from flow_aware_encoders.flowerformer import FLOWERFormer
from train_utils import train_dict
from src import *

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], idx  # Return index along with data and label

if __name__ == "__main__" : 
    
    warnings.filterwarnings("ignore")
    
    parser = argparse.ArgumentParser('Proposed Method.')
    
    parser.add_argument('-dname', '--dname', type=str, default='nb101')            
    parser.add_argument('-device', '--device', type=str, default='cuda:0')
    parser.add_argument('-train_ratio', '--train_ratio', type=float, default=0.01)
    parser.add_argument('-proj_layer', '--proj_layer', type=int, default=2)
    parser.add_argument('-proj_dim1', '--proj_dim1', type=int, default=128)
    parser.add_argument('-proj_dim2', '--proj_dim2', type=int, default=64)
    parser.add_argument('-ssl_lr', '--ssl_lr', type=float, default=1e-3)
    parser.add_argument('-wdecay', '--wdecay', type=float, default=1e-6)
    parser.add_argument('-lamda1', '--lamda1', type=float, default=0.5)
    parser.add_argument('-lamda2', '--lamda2', type=float, default=0.5)
    parser.add_argument('-enc', '--enc', type=str, default="gnn")
    
    args = parser.parse_args()
    
    dname = args.dname
    device = args.device
    train_ratio = args.train_ratio
    proj_dim1 = args.proj_dim1
    proj_dim2 = args.proj_dim2
    proj_layer = args.proj_layer
    encoder_type = args.enc
    gnn_type = args.enc
    ssl_lr = args.ssl_lr
    wdecay = args.wdecay
    lamda1 = args.lamda1
    lamda2 = args.lamda2

    if dname == "nb101" : 
        n_train = 7290

    elif dname == "nb201" : 
        n_train = 7813

    elif dname == "nb301" : 
        n_train = 5611

    else : 
        raise TypeError("Data should be given one of (nb101, nb201, nb301).")
    
    data = torch.load("./dataset/{0}_unified.pt".format(dname))
    d_proxy = data[0].fwbw.shape[1]
    data_down = copy.deepcopy(data)
    n_feat = data_down[0].x.shape[1]
    
    if encoder_type in ["gnn", "gin"] : # GatedGCN        
        BSize = 256
        
        for d in data :  # Make the graph as undirected
            d.edge_index = torch.hstack([d.edge_index, d.edge_index[[1, 0], :]])
            
        for d_new in data_down : 
            d_new.edge_index = torch.hstack([d_new.edge_index, d_new.edge_index[[1, 0], :]])
            
        param_lists = {"nb101" : {"hidden_dim" : 256, "lr" : 0.001, "n_layers" : 3, "epochs" : 200},
                        "nb201" : {"hidden_dim" : 128,  "lr" : 0.001, "n_layers" : 3, "epochs" : 500},        
                        "nb301" : {"hidden_dim" : 256, "lr" : 0.001, "n_layers" : 3, "epochs" : 300}}
            
    else : # FlowerFormer
        BSize = 1024
  
        param_lists = {"nb101" : {"hidden_dim" : 64, "lr" : 0.001, "n_layers" : 3, "epochs" : 300},  
                      "nb201" : {"hidden_dim" : 64,  "lr" : 0.001, "n_layers" : 3, "epochs" : 500}, 
                      "nb301" : {"hidden_dim" : 256, "lr" : 0.001, "n_layers" : 2, "epochs" : 300}}
            
    data = MyDataset(data)
    data_down = MyDataset(data_down)
    
    edge_labels = []
    
    hid_dim = param_lists[dname]["hidden_dim"]
    n_layers = param_lists[dname]["n_layers"]
    down_lr = param_lists[dname]["lr"]
    down_epochs = param_lists[dname]["epochs"]
    
    perms, _ = get_perms([21, 2021, 202121], len(data_down), False)
    
    with open("./flow_aware_encoders/FLOWERFormer_{0}.yaml".format(dname)) as f:
        yaml_object = yaml.safe_load(f)
        
    cfg = Option(yaml_object)
    
    results = {train_ratio: {"kt": [], "sp": [], "pak": []} for train_ratio in [train_ratio]}
    all_results = dict()

    TK = 0.0
    TP = 0.0
    
    totals = dict()
    
    """
    Let's start pre-training.
    """
    hyp_lists = "{0}_{1}_{2}_{3}_{4}".format(ssl_lr, wdecay, proj_dim1, proj_dim2, proj_layer)
    print(hyp_lists)

    ## The same seed leveraged in (Hwang et al., CVPR 2024) and (Ning et al., NeurIPS 2022)
    for seed in [21, 2021, 202121] :  ## Random seed

        for perm in perms : ## Random dataset splot

            seed_everything(seed) # Fix seed

            if encoder_type == "gatedgcn" : # If GatedGCN
                model = GatedGCNModel(in_channels = n_feat, hid_channels = hid_dim, 
                                  num_layers = n_layers, dp = 0.1).to(device) ## 201 is layer 3
                proj_in_dim = hid_dim

            elif encoder_type == "gin" : # If GIN
                model = GINModel(in_channels = n_feat, hid_channels = hid_dim, 
                                  num_layers = n_layers, dp = 0.1).to(device) ## 201 is layer 3
                proj_in_dim = int(hid_dim * n_layers)

            elif encoder_type == "flowerformer" : # If FlowerFormer
                model = FLOWERFormer(cfg.dag.dim_in, 1, cfg).to(device) 
                proj_in_dim = hid_dim
            
            else : 
                raise TypeError("Encoder should be given one of (gatedgcn, gin, flowerformer)")

            flow_decoder = ProjectionHead(proj_in_dim, proj_dim1, d_proxy, proj_layer).to(device) # Projection for flow
            proxy_decoder = ProjectionHead(proj_in_dim, proj_dim2, 1, proj_layer).to(device) # Projection for surrogate
            
            train_func, eval_func = train_dict["1cell"]

            SSL_train_loader, _, _ = get_data(data, perm = perm, batch_size = BSize, train_ratio = 1.0, num_train = len(data)) ## 1024
            
            model = FGP(encoder = model, flow_decoder = flow_decoder, proxy_decoder = proxy_decoder, 
                        lr = ssl_lr, epochs = 200, loader = SSL_train_loader, device = device, cfg = cfg, 
                        l1 = lamda1, l2 = lamda2, w_decay = wdecay)

            if encoder_type in ["gatedgcn", "gin"] : # If GatedGCN

                optimizer = torch.optim.AdamW(model.parameters(), lr=down_lr, weight_decay=1e-6)

                train_loader, val_loader, test_loader = get_data(data_down, perm = perm, 
                                                            batch_size = 16, train_ratio = train_ratio, 
                                                           num_train = n_train)

            else :

                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optim.base_lr, weight_decay=cfg.optim.weight_decay)
                scheduler = cosine_with_warmup_scheduler(optimizer, cfg.optim.num_warmup_epochs, cfg.optim.max_epoch)


                train_loader, val_loader, test_loader = get_data(data_down, perm = perm, 
                                                            batch_size = 128, train_ratio = train_ratio, 
                                                           num_train = n_train)

            best_val_kt = 0
            best_val_model = copy.deepcopy(model.state_dict())

            for epoch in range(down_epochs):

                train_loss = train_func(model, train_loader, optimizer, device, cfg)
                val_metrics = eval_func(model, val_loader, device)

                if val_metrics["kt"] > best_val_kt:
                    best_val_kt = val_metrics["kt"]
                    best_val_model = copy.deepcopy(model.state_dict())

                if encoder_type in ["gatedgcn", "gin"] : # If GatedGCN
                    None
                else: 
                    scheduler.step()

            model.load_state_dict(best_val_model)

            test_metrics = eval_func(model, test_loader, device)
            results[train_ratio]["kt"].append(test_metrics["kt"])
            results[train_ratio]["sp"].append(test_metrics["sp"])
            results[train_ratio]["pak"].append(test_metrics["pak"])
            pak_unpack = [
                test_metrics["pak"][ratio][3] for ratio in [0.01, 0.05, 0.1, 0.5, 1.0]
            ]
            print("Test KT: {}, Test P@k: {}".format(test_metrics["kt"], pak_unpack))