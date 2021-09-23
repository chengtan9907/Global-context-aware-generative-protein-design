from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from .utils import _dihedrals, _hbonds, _rbf, \
                   _orientations_coarse, _contacts, _dist
from .self_attention import Normalize

# Thanks for StructTrans
# https://github.com/jingraham/neurips19-graph-protein-design
class PositionalEncodings(nn.Module):
    def __init__(self, num_embeddings, period_range=[2,1000]):
        super(PositionalEncodings, self).__init__()
        self.num_embeddings = num_embeddings
        self.period_range = period_range 

    def forward(self, E_idx):
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1)).to(E_idx.device)
        d = (E_idx.float() - ii).unsqueeze(-1)
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        ).to(E_idx.device)
        angles = d * frequency.view((1,1,1,-1))
        return torch.cat((torch.cos(angles), torch.sin(angles)), -1)

class ProteinFeatures(nn.Module):
    def __init__(self, edge_features, node_features, num_positional_embeddings=16,
        num_rbf=16, top_k=30, features_type='full', augment_eps=0., dropout=0.1):
        super(ProteinFeatures, self).__init__()
        self.edge_features = edge_features
        self.node_features = node_features
        self.top_k = top_k
        self.augment_eps = augment_eps 
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings

        self.features_type = features_type
        self.feature_dimensions = {
            'coarse': (3, num_positional_embeddings + num_rbf + 7),
            'full': (6, num_positional_embeddings + num_rbf + 7),
            'dist': (6, num_positional_embeddings + num_rbf),
            'hbonds': (3, 2 * num_positional_embeddings),
        }

        self.embeddings = PositionalEncodings(num_positional_embeddings)
        self.dropout = nn.Dropout(dropout)
        
        node_in, edge_in = self.feature_dimensions[features_type]
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

    def forward(self, X, mask, **kwargs):
        if self.training and self.augment_eps > 0:
            X = X + self.augment_eps * torch.randn_like(X)
        X_ca = X[:,:,1,:]
        D_neighbors, E_idx, mask_neighbors = _dist(X_ca, mask, self.top_k)

        AD_features, O_features = _orientations_coarse(X_ca, E_idx)
        RBF = _rbf(D_neighbors, self.num_rbf)
        E_positional = self.embeddings(E_idx)
        
        if self.features_type == 'coarse':
            V = AD_features
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'hbonds':
            neighbor_C, neighbor_HB = _contacts(D_neighbors, mask_neighbors), \
                                      _hbonds(X, E_idx, mask_neighbors)
            neighbor_C, neighbor_HB = self.dropout(neighbor_C), self.dropout(neighbor_HB)
            V = mask.unsqueeze(-1) * torch.ones_like(AD_features)
            neighbor_C = neighbor_C.expand(-1,-1,-1, int(self.num_positional_embeddings / 2))
            neighbor_HB = neighbor_HB.expand(-1,-1,-1, int(self.num_positional_embeddings / 2))
            E = torch.cat((E_positional, neighbor_C, neighbor_HB), -1)
        elif self.features_type == 'full':
            V = _dihedrals(X)
            E = torch.cat((E_positional, RBF, O_features), -1)
        elif self.features_type == 'dist':
            V = _dihedrals(X)
            E = torch.cat((E_positional, RBF), -1)

        V = self.norm_nodes(self.node_embedding(V))
        E = self.norm_edges(self.edge_embedding(E))
        return V, E, E_idx