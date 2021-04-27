import torch.nn as nn
import torch.nn.functional as F


class MLP_follow_model(nn.Module):

    def __init__(self, mdoel, h_dim, out_dim):
        self.gnn_model = mdoel
        self.MLP = nn.Linear(h_dim, out_dim)

    def forward(self, hg, h=None):
        if h is None:
            h = self.gnn_model(hg)
        else:
            h = self.gnn_model(hg, h)
        h = self.MLP(h)
        return h




