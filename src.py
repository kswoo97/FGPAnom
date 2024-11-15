import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from tqdm import trange

import math
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from torch_scatter import scatter_sum

import copy

from scipy.stats import stats
from utils import test_xk

from torch_geometric.nn import GINConv, ResGatedGraphConv
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Data

## Many parts of our code is from Hwang et al., CVPR 2024.
## Official implementation of the above paper is in https://github.com/y0ngjaenius/CVPR2024_FLOWERFormer


class ProjectionHead(torch.nn.Module) : 
    
    def __init__(self, in_dim, hid_dim, nC, n_layers = 2) : 
        
        super(ProjectionHead, self).__init__()
        
        self.n_layers = n_layers
        
        if self.n_layers == 1 : 
            self.proj_head1 = torch.nn.Linear(in_dim, nC) # Originally 32
            
        elif self.n_layers == 2 : 
            self.proj_head1 = torch.nn.Linear(in_dim, hid_dim) # Originally 32
            self.proj_head2 = torch.nn.Linear(hid_dim, nC) # Originally 32
            
        else : 
            self.proj_head1 = torch.nn.Linear(in_dim, hid_dim) # Originally 32
            self.proj_head2 = torch.nn.Linear(hid_dim, int(hid_dim/2)) # Originally 32
            self.proj_head3 = torch.nn.Linear(int(hid_dim/2), nC) # Originally 32
        
    def forward(self, x) : 
        
        if self.n_layers == 1 : 
            x = self.proj_head1(x)
            
        elif self.n_layers == 2 : 
            x = self.proj_head1(x)
            x = torch.relu(x)
            x = self.proj_head2(x)
        
        else : 
            x = self.proj_head1(x)
            x = torch.relu(x)
            x = self.proj_head2(x)
            x = torch.relu(x)
            x = self.proj_head3(x)

        return x
    
    
def FGP(encoder, flow_decoder, proxy_decoder, 
        lr, epochs, loader, device, cfg, 
        l1 = 0.5, l2 = 0.5, w_decay = 1e-6) :


    optimizer = torch.optim.AdamW(list(encoder.parameters()) 
                                   + list(flow_decoder.parameters()) 
                                   + list(proxy_decoder.parameters()),
                                 lr = lr, weight_decay = w_decay) # 1e-4

    def give_proxy_loss(cfg, y, sample_scores) : 
        
        sample_scores = sample_scores.squeeze()
        
        if sample_scores.ndim == 0:
            return 0.0
        
        n_max_pairs = int(cfg.dag.compare.max_compare_ratio * len(batch))
        y = y.cpu().detach().numpy()
        acc_diff = y[:, None] - y
        acc_abs_diff_matrix = np.triu(np.abs(acc_diff), 1)
        ex_thresh_inds = np.where(acc_abs_diff_matrix > 0.0)
        ex_thresh_num = len(ex_thresh_inds[0])

        if ex_thresh_num > n_max_pairs:
            keep_inds = np.random.choice(np.arange(ex_thresh_num), n_max_pairs, replace=False)
            ex_thresh_inds = (ex_thresh_inds[0][keep_inds], ex_thresh_inds[1][keep_inds])

        better_labels = (acc_diff > 0)[ex_thresh_inds]
        n_diff_pairs = len(better_labels)

        s_1 = sample_scores[ex_thresh_inds[1]]
        s_2 = sample_scores[ex_thresh_inds[0]]

        better_pm = 2 * s_1.new(np.array(better_labels, dtype=np.float32)) - 1
        zero_, margin = s_1.new([0.0]), s_1.new([cfg.dag.compare.margin])
        
        loss = torch.mean(torch.max(zero_, margin - better_pm * (s_2 - s_1)))
        
        return loss
    
    def give_recon_loss(x_answer, x_gen) : 
        
        return torch.mean((torch.sum((x_answer - x_gen) ** 2 , dim = 1)))
    
    encoder.train()
    
    flow_decoder.train()
    proxy_decoder.train()

    for epoch in trange(epochs) : ## First round training
        
        for batch, _ in loader : 

            batch = batch.to(device)
            
            optimizer.zero_grad()
            
            Vembs = encoder.node_embedding(batch) ## Return node embeddings
            Gembs = global_mean_pool(Vembs, batch.batch) ## Return graph embeddings
            
            
            genX = flow_decoder(Gembs)
            genP = proxy_decoder(Gembs)
            
            origX = batch.fwbw
            origP = batch.proxy1
            
            loss1 = give_recon_loss(genX, origX)
            loss2 = give_proxy_loss(cfg, origP, genP)
            loss = l1 * loss1 + l2 * loss2

            loss.backward()
            optimizer.step()

    return encoder


## Non-flow-aware Neural Architecture Encoders

    
class MLP(torch.nn.Module):
    def __init__(self, num_features, num_classes, hidden_units=32, num_layers=1, bias_term = True):
        super(MLP, self).__init__()
        if num_layers == 1:
            self.layers = nn.Linear(num_features, num_classes, bias = bias_term)
        elif num_layers > 1:
            layers = [nn.Linear(num_features, hidden_units, bias = bias_term),
                      #nn.BatchNorm1d(hidden_units),
                      nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_units, hidden_units, bias = bias_term),
                               #nn.BatchNorm1d(hidden_units),
                               nn.ReLU()])
            layers.append(nn.Linear(hidden_units, num_classes, bias = bias_term))
            self.layers = nn.Sequential(*layers)
        else:
            raise ValueError()

    def forward(self, x):
        return self.layers(x)

class GINModel(torch.nn.Module):
    
    def __init__(self, in_channels, hid_channels=32, num_layers=3, dp=0.15):
        super(GINModel, self).__init__()
        
        convs, bns = [], []
        
        for i in range(num_layers):
            input_dim = in_channels if i == 0 else hid_channels
            hidden_dim = hid_channels
            convs.append(GINConv(MLP(input_dim, hidden_dim, hidden_dim, 2)))
            bns.append(nn.BatchNorm1d(hidden_dim))
            
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.num_layers = num_layers        
        self.dropout_layer = torch.nn.Dropout(p = dp)
        self.last_linear_layer1 = torch.nn.Linear(int(num_layers * hidden_dim), hidden_dim)
        self.last_linear_layer2 = torch.nn.Linear(int(hidden_dim), 1)
        
    def reset_parameters(self):
        for conv in self.convs: conv.reset_parameters()
        
    def reset_linear_layers(self) : 
        
        self.last_linear_layer1.reset_parameters()
        self.last_linear_layer2.reset_parameters()
            
    def forward(self, data) :
        
        x = data.x 
        edge_index = data.edge_index
        
        h_list = [x]
        
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h_list[-1], edge_index)
            h = self.dropout_layer(h)
            h = torch.relu(h)
            h_list.append(h)
        
        out = torch.cat(h_list[1:], 1)
        graph_embs = global_mean_pool(out, data.batch)
        
        graph_y = self.last_linear_layer1(graph_embs)
        graph_y = torch.relu(graph_y)
        graph_y = self.last_linear_layer2(graph_y)
        
        return graph_y, data.y
    
    def node_embedding(self, data) :
        
        x = data.x 
        edge_index = data.edge_index
        
        h_list = [x] # Do not add its original feature
        
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h_list[-1], edge_index)
            h = self.dropout_layer(h)
            h = torch.relu(h)
            h_list.append(h)
        
        out = torch.cat(h_list[1:], 1)
        
        return out
    
class GatedGCNModel(MessagePassing) :
    
    def __init__(self, in_channels, hid_channels, num_layers, dp):
        super().__init__()
        self.in_channels = in_channels
        self.hid_channels = hid_channels
        self.dp = dp
        self.num_layers = num_layers
        self.setup_layers()
        self.reset_parameters()
        
    def setup_layers(self):
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.in_layers = torch.nn.Linear(self.in_channels, self.hid_channels)
        # inchannels = self.in_channels
        inchannels = self.hid_channels
        for i in range(self.num_layers-1):
            self.convs.append(ResGatedGraphConv(inchannels, self.hid_channels))
            inchannels = self.hid_channels
        self.convs.append(ResGatedGraphConv(inchannels, self.hid_channels))
        
        self.last_linear_layer1 = torch.nn.Linear(self.hid_channels, self.hid_channels)
        self.last_linear_layer2 = torch.nn.Linear(self.hid_channels, 1)
        
        self.dropout = nn.Dropout(self.dp)
        
    def reset_parameters(self):
        for conv in self.convs: conv.reset_parameters()
        
    def reset_linear_layers(self) : 
        
        self.last_linear_layer1.reset_parameters()
        self.last_linear_layer2.reset_parameters()
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x0 = self.in_layers(x)
        
        x = x0.clone()
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = 0.2 * x0 + 0.8 * x
            x = torch.relu(x)
            x = self.dropout(x)
        
        graph_embs = global_mean_pool(x, data.batch)
        
        graph_y = self.last_linear_layer1(graph_embs)
        graph_y = torch.relu(graph_y)
        graph_y = self.last_linear_layer2(graph_y)
        
        return graph_y, data.y
    
    def node_embedding(self, data) :
        
        x, edge_index = data.x, data.edge_index
        x0 = self.in_layers(x)
        x = x0.clone()
        
        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = 0.2 * x0 + 0.8 * x
            x = torch.relu(x)
            x = self.dropout(x)

        return x