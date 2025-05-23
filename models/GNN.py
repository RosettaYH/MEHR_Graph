# Reference: https://github.com/HKU-MedAI/MulT-EHR

from torch import nn

class GNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, n_layers, activation, dropout, tasks):
        super().__init__()

        self.n_layers = n_layers
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.activation = activation
        self.dor = dropout
        self.dropout = nn.Dropout(p=dropout)

        self.tasks = tasks

        self.layers = self.get_layers()

        self.out = nn.ModuleDict()
        for t in tasks:
            if t in ["readm", "mort_pred"]:
                self.out[t] = nn.Linear(hidden_dim, 2)
            elif t == "drug_rec":
                self.out[t] = nn.Linear(hidden_dim, out_dim)
        self.embeddings = None

    def get_layers(self):
        raise NotImplementedError

    def get_logit(self, g, h, causal=False):
        raise NotImplementedError

    def set_embeddings(self, emb):
        self.embeddings = emb

    