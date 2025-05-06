# Reference: https://github.com/HKU-MedAI/MulT-EHR

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData

from .GNN import GNN


class HGTConv(nn.Module):
    def __init__(self, in_dim, out_dim, num_types, num_relations, n_heads, dropout=0.2, use_norm=False):
        super(HGTConv, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_types = num_types
        self.num_relations = num_relations
        self.n_heads = n_heads
        self.d_k = out_dim // n_heads
        self.sqrt_dk = math.sqrt(self.d_k)
        self.use_norm = use_norm

        self.k_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.q_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.v_linears = nn.ModuleList([nn.Linear(in_dim, out_dim) for _ in range(num_types)])
        self.a_linears = nn.ModuleList([nn.Linear(out_dim, out_dim) for _ in range(num_types)])
        self.norms = nn.ModuleList([nn.LayerNorm(out_dim) for _ in range(num_types)])

        self.relation_pri = nn.Parameter(torch.ones(num_relations, n_heads))
        self.relation_att = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))
        self.relation_msg = nn.Parameter(torch.Tensor(num_relations, n_heads, self.d_k, self.d_k))

        self.skip = nn.Parameter(torch.ones(num_types))
        self.drop = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.relation_att)
        nn.init.xavier_uniform_(self.relation_msg)

    def forward(self, x_dict, edge_index_dict, node_type_dict, edge_type_dict):
        for ntype, x in x_dict.items():
            tid = node_type_dict[ntype]
            x_dict[ntype] = {
                'k': self.k_linears[tid](x).view(-1, self.n_heads, self.d_k),
                'v': self.v_linears[tid](x).view(-1, self.n_heads, self.d_k),
                'q': self.q_linears[tid](x).view(-1, self.n_heads, self.d_k),
                'inp': x
            }

        out_dict = {}

        for (srctype, rel, dsttype), edge_index in edge_index_dict.items():
            rid = edge_type_dict[rel]
            sid = node_type_dict[srctype]
            did = node_type_dict[dsttype]

            src_idx, dst_idx = edge_index

            k = x_dict[srctype]['k'][src_idx] 
            v = x_dict[srctype]['v'][src_idx]
            q = x_dict[dsttype]['q'][dst_idx]

            R_att = self.relation_att[rid]
            R_msg = self.relation_msg[rid]
            R_pri = self.relation_pri[rid]

            k = torch.einsum('ehd,hdf->ehf', k, R_att)
            v = torch.einsum('ehd,hdf->ehf', v, R_msg)

            att = (q * k).sum(dim=-1) * R_pri / self.sqrt_dk 
            att = F.softmax(att, dim=0)

            out = torch.sum(att.unsqueeze(-1) * v, dim=0)  
            out = out.reshape(-1)                       

            out = self.a_linears[did](out)

            alpha = torch.sigmoid(self.skip[did])
            inp = x_dict[dsttype]['inp']
            out = out * alpha + inp * (1 - alpha)

            if self.use_norm:
                out = self.norms[did](out)

            out_dict[dsttype] = self.drop(out)

        return out_dict

class HGT(GNN):
    def __init__(self, G, n_inp, n_hid, n_out, n_layers, n_heads, tasks, dropout, use_norm=True):
        self.n_heads = n_heads
        self.use_norm = use_norm
        self.ntypes = G.node_types
        self.etypes = list(G.edge_index_dict.keys())

        self.node_type_dict = {ntype: i for i, ntype in enumerate(self.ntypes)}
        self.edge_type_dict = {rel: i for i, rel in enumerate(set([r for (_, r, _) in self.etypes]))}

        super(HGT, self).__init__(n_inp, n_hid, n_out, n_layers, F.relu, dropout, tasks)

        self.adapt_ws = nn.ModuleList([
            nn.Linear(n_inp, n_hid) for _ in self.ntypes
        ])

        self.out = nn.ModuleDict()
        for t in tasks:
            if t == "drug_rec":
                self.out[t] = nn.Linear(n_hid, n_out)
            elif t == "los":
                self.out[t] = nn.Linear(n_hid, 10)
            else:
                self.out[t] = nn.Linear(n_hid, 2)

    def get_layers(self):
        return nn.ModuleList([
            HGTConv(
                in_dim=self.hidden_dim,
                out_dim=self.hidden_dim,
                num_types=len(self.ntypes),
                num_relations=len(self.edge_type_dict),
                n_heads=self.n_heads,
                dropout=self.dor,
                use_norm=self.use_norm
            )
            for _ in range(self.n_layers)
        ])

    def get_logit(self, data: HeteroData, h=None):
        layers = self.layers

        h_dict = {}
        for i, ntype in enumerate(self.ntypes):
            x = data[ntype].feat if hasattr(data[ntype], 'feat') else data[ntype].x
            h_dict[ntype] = F.gelu(self.adapt_ws[i](x))

        for layer in layers:
            h_dict = layer(h_dict, data.edge_index_dict, self.node_type_dict, self.edge_type_dict)

        return h_dict

    def forward(self, data: HeteroData, out_key: str, task: str):
        logits = self.get_logit(data)
        self.embeddings = torch.cat(list(logits.values()))
        out = self.out[task](logits[out_key])
        return out