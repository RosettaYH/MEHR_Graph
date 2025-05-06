import torch.nn.functional as F
from models import GCN, GAT, HGT
import torch
from pyhealth.metrics import binary_metrics_fn, multilabel_metrics_fn, multiclass_metrics_fn

import numpy as np
import pickle

import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from collections import OrderedDict
from sklearn.model_selection import train_test_split


seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def ordered_yaml():
    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper


def load_graph(graph_path, labels_path, feat_dim=128):
    with open(graph_path, 'rb') as inp:
        unp = pickle.Unpickler(inp)
        g = unp.load()

    with open(labels_path, 'rb') as inp:
        unp = pickle.Unpickler(inp)
        labels = unp.load()

    for tp in g.node_types:
        n_nodes = g[tp].num_nodes  
        feat = torch.randn(n_nodes, feat_dim)
        g[tp].x = feat

    train_masks = {}
    test_masks = {}
    for k, lb in labels.items():
        if k == "all_drugs":
            train_masks[k] = lb
            test_masks[k] = lb
            continue

        all_visits = np.array(list(lb.keys()))

        if k != "drug_rec":
            all_labels = np.array([lb[v] for v in all_visits])
            train_visits, test_visits = train_test_split(
                all_visits,
                test_size=0.1,
                stratify=all_labels,
                random_state=42
            )
        else:
            indices = np.random.permutation(len(all_visits))
            split = int(0.9 * len(indices))
            train_visits = all_visits[indices[:split]]
            test_visits = all_visits[indices[split:]]

        train_masks[k] = train_visits
        test_masks[k] = test_visits

    return g, labels, train_masks, test_masks

def parse_gnn_model(config_gnn, g, tasks=None):
    gnn_name = config_gnn["name"]
    if gnn_name == "GCN":
        return GCN(
            in_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            n_layers=config_gnn["num_layers"],
            activation=F.relu,
            dropout=config_gnn["feat_drop"],
            tasks=tasks,
        )
    elif gnn_name == "GAT":
        n_layers = config_gnn["num_layers"]
        n_heads = config_gnn["num_heads"]
        n_out_heads = config_gnn["num_out_heads"]
        heads = ([n_heads] * n_layers) + [n_out_heads]
        return GAT(
            n_layers=config_gnn["num_layers"],
            in_dim=config_gnn["in_dim"],
            hidden_dim=config_gnn["hidden_dim"],
            out_dim=config_gnn["out_dim"],
            heads=heads,
            activation=F.leaky_relu,
            feat_drop=config_gnn["feat_drop"],
            attn_drop=config_gnn["attn_drop"],
            negative_slope=config_gnn["negative_slope"],
            tasks=tasks,
            residual=False
        )
    elif gnn_name == "HGT":
        return HGT(
            g,
            n_inp=config_gnn["in_dim"],
            n_hid=config_gnn["hidden_dim"],
            n_out=config_gnn["out_dim"],
            n_layers=config_gnn["num_layers"],
            n_heads=config_gnn["num_heads"],
            dropout=config_gnn["feat_drop"],
            tasks=tasks,
        )
    else:
        raise NotImplementedError("Model is not implemented")


def add_reverse_edges(data):
    new_data = data.clone()
    for edge_type in list(data.edge_types):
        src, rel, dst = edge_type
        edge_index = data[edge_type].edge_index
        rev_edge_type = (dst, rel + "_rev", src)
        new_data[rev_edge_type].edge_index = edge_index.flip(0)
    return new_data


def hetero_subgraph(data, node_subset):
    new_data = type(data)()
    node_mappings = {}
    for nt in data.node_types:
        if nt in node_subset:
            nodes = node_subset[nt]
        else:
            nodes = torch.arange(data[nt].num_nodes)
        mapping = {int(old.item()): int(new) for new, old in enumerate(nodes)}
        node_mappings[nt] = mapping
        new_data[nt].num_nodes = len(nodes)
        if 'x' in data[nt]:
            new_data[nt].x = data[nt].x[nodes]
    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        edge_index = data[edge_type].edge_index
        src_map = node_mappings[src]
        dst_map = node_mappings[dst]

        mask_src = torch.tensor([int(idx.item()) in src_map for idx in edge_index[0]])
        mask_dst = torch.tensor([int(idx.item()) in dst_map for idx in edge_index[1]])
        mask = mask_src & mask_dst
        if mask.sum() == 0:
            continue
        filtered_edge_index = edge_index[:, mask]
        new_src = torch.tensor([src_map[int(idx.item())] for idx in filtered_edge_index[0]])
        new_dst = torch.tensor([dst_map[int(idx.item())] for idx in filtered_edge_index[1]])
        new_data[edge_type].edge_index = torch.stack([new_src, new_dst], dim=0)
    return new_data

def metrics(outputs, targets, t, prefix="train"):

    if t in ["mort_pred", "readm"]:
        met = binary_metrics_fn(
            targets.detach().cpu().numpy(),
            outputs.softmax(1)[:, 1].detach().cpu().numpy(),
            metrics=["accuracy", "roc_auc", "f1", "pr_auc"]
        )
    elif t == "drug_rec":
        met = multilabel_metrics_fn(
            targets.detach().cpu().numpy(),
            outputs.detach().cpu().numpy(),
            metrics=["roc_auc_samples", "pr_auc_samples", "accuracy", "f1_weighted", "jaccard_weighted"]
        )
    else:
        raise ValueError

    met = {f"{prefix}_{k}": v for k, v in met.items()}
    return met
