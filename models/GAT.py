# Reference: https://github.com/HKU-MedAI/MulT-EHR

import torch
import torch.nn as nn
from torch_geometric.nn import GATConv
from torch_geometric.data import HeteroData
from torch_geometric.utils import add_self_loops

from .GNN import GNN

class GAT(GNN):
    def __init__(self, n_layers, in_dim, hidden_dim, out_dim, heads, activation, 
                 feat_drop, attn_drop, negative_slope, residual, tasks):
        self.heads = heads
        self.attn_drop = attn_drop
        self.negative_slope = negative_slope
        self.residual = residual

        super().__init__(in_dim, hidden_dim, out_dim, n_layers, activation, feat_drop, tasks)

    def forward(self, data: HeteroData, nt, task):
        data_hom = data.to_homogeneous(node_attrs=["x"])
        data_hom.edge_index, _ = add_self_loops(
            data_hom.edge_index, num_nodes=data_hom.num_nodes
        )
        
        x = data_hom.x
        
        logits = self.get_logit(data_hom, x)
        out_all = self.out[task](logits)
        
        visit_idx = data.node_types.index("visit")
        mask = data_hom.node_type == visit_idx
        out = out_all[mask]
        self.set_embeddings(logits)
        
        return out
    
    def get_logit(self, g, h, causal=False):
        layers = self.layers if not causal else self.rand_layers
        
        for i, layer in enumerate(layers):
            if i != 0:
                h = self.dropout(h)
            
            h = layer(h, g.edge_index)
            
            if i != len(layers) - 1:
                h = h.flatten(1)
        self.embeddings = h
        return h
    
    def get_layers(self):
        layers = nn.ModuleList()
        for l in range(self.n_layers):
            in_ch = self.in_dim if l == 0 else self.hidden_dim * self.heads[l - 1]
            out_ch = self.hidden_dim
            num_heads = self.heads[l]
            concat = True if l != self.n_layers - 1 else False 

            layers.append(GATConv(
                in_ch, out_ch, heads=num_heads,
                concat=concat,
                dropout=self.attn_drop,
                negative_slope=self.negative_slope,
                add_self_loops=True,
                bias=True
            ))
        return layers
