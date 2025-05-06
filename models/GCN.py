# Reference: https://github.com/HKU-MedAI/MulT-EHR

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_self_loops

from .GNN import GNN

class GCN(GNN):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, activation, dropout, tasks):
        super().__init__(in_dim, hidden_dim, out_dim, n_layers, activation, dropout, tasks)
        self.n_layers = n_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.activation = activation
        self.layers = self.get_layers()

    def forward(self, data, nt, task):
        data_hom = data.to_homogeneous(node_attrs=['x'])
        edge_index, _ = add_self_loops(data_hom.edge_index, num_nodes=data_hom.num_nodes)
        data_hom.edge_index = edge_index

        x = data_hom.x
        logits = self.get_logit(data_hom, x)
        out_all = self.out[task](logits)
        
        visit_idx = data.node_types.index("visit")
        mask = data_hom.node_type == visit_idx
        out = out_all[mask]

        return out


    def get_logit(self, data_hom, x, causal=False):
        layers = self.layers if not causal else self.rand_layers
        edge_index = data_hom.edge_index
        h = x
        for i, layer in enumerate(layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(h, edge_index)
            if self.activation is not None:
                h = self.activation(h)
        self.set_embeddings(h)
        return h

    def get_layers(self):
        layers = nn.ModuleList()
        layers.append(GCNConv(self.in_dim, self.hidden_dim, add_self_loops=False))
        for i in range(self.n_layers - 1):
            layers.append(GCNConv(self.hidden_dim, self.hidden_dim, add_self_loops=False))
        return layers
