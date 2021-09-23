import torch
import torch.nn as nn
import torch.nn.functional as F
from .self_attention import NeighborAttention as Attention
from .self_attention import Normalize, PositionWiseFeedForward
from .protein_features import PositionalEncodings
from .utils import gather_nodes, cat_neighbors_nodes, \
                   _dihedrals, _rbf, _orientations_coarse_gl

class Local_Module(nn.Module):
    def __init__(self, num_hidden, num_in, is_attention, dropout=0.1, scale=30):
        super(Local_Module, self).__init__()
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.is_attention = is_attention
        self.scale = scale
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])
        self.W = nn.Sequential(*[
            nn.Linear(num_hidden + num_in, num_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden, num_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden, num_hidden)
        ])
        self.A = nn.Parameter(torch.empty(size=(num_hidden + num_in, 1)))
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        # Concatenate h_V_i to h_E_ij
        h_V_expand = h_V.unsqueeze(-2).expand(-1,-1,h_E.size(-2),-1)
        h_EV = torch.cat([h_V_expand, h_E], -1)
        # get message
        h_message = self.W(h_EV)
        # Attention
        if self.is_attention == 1:
            e = F.sigmoid(F.leaky_relu(torch.matmul(h_EV, self.A))).squeeze(-1).exp()
            e = e / e.sum(-1).unsqueeze(-1)
            h_message = h_message * e.unsqueeze(-1)

        if mask_attend is not None:
            h_message = mask_attend.unsqueeze(-1) * h_message
        dh = torch.sum(h_message, -2) / self.scale

        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))

        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class Global_Module(nn.Module):
    def __init__(self, num_hidden, num_in, num_heads=4, dropout=0.1):
        super(Global_Module, self).__init__()
        self.num_heads = num_heads
        self.num_hidden = num_hidden
        self.num_in = num_in
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.ModuleList([Normalize(num_hidden) for _ in range(2)])
        self.attention = Attention(num_hidden, num_in, num_heads)
        self.dense = PositionWiseFeedForward(num_hidden, num_hidden * 4)

    def forward(self, h_V, h_E, mask_V=None, mask_attend=None):
        dh = self.attention(h_V, h_E, mask_attend)
        h_V = self.norm[0](h_V + self.dropout(dh))
        dh = self.dense(h_V)
        h_V = self.norm[1](h_V + self.dropout(dh))
        if mask_V is not None:
            mask_V = mask_V.unsqueeze(-1)
            h_V = mask_V * h_V
        return h_V


class GCA(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, 
        num_encoder_layers=3, num_decoder_layers=3, vocab=20, 
        k_neighbors=30, dropout=0.1, is_attention=0, **kwargs):
        """ Graph labeling network """
        super(GCA, self).__init__()
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.top_k = k_neighbors
        self.num_rbf = 16
        self.num_positional_embeddings = 16

        node_in, edge_in = 6, 39 - 16
        self.embeddings = PositionalEncodings(self.num_positional_embeddings)
        self.node_embedding = nn.Linear(node_in, node_features, bias=True)
        self.edge_embedding = nn.Linear(edge_in, edge_features, bias=True)
        self.norm_nodes = Normalize(node_features)
        self.norm_edges = Normalize(edge_features)

        self.W_v = nn.Linear(node_features, hidden_dim, bias=True)
        self.W_e = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_f = nn.Linear(edge_features, hidden_dim, bias=True)
        self.W_s = nn.Embedding(vocab, hidden_dim)

        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            self.encoder_layers.append(nn.ModuleList([
                Local_Module(hidden_dim, hidden_dim*2, is_attention=is_attention, dropout=dropout),
                Global_Module(hidden_dim, hidden_dim*2, dropout=dropout)
            ]))

        self.decoder_layers = nn.ModuleList([])
        for _ in range(num_decoder_layers):
            self.decoder_layers.append(
                Local_Module(hidden_dim, hidden_dim*3, is_attention=is_attention, dropout=dropout)
            )

        self.W_out = nn.Linear(hidden_dim, vocab, bias=True)
        self._init_params()

    def _init_params(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _autoregressive_mask(self, E_idx):
        N_nodes = E_idx.size(1)
        ii = torch.arange(N_nodes)
        ii = ii.view((1, -1, 1)).to(E_idx.device)
        mask = E_idx - ii < 0
        mask = mask.type(torch.float32)
        return mask

    def _get_encoder_mask(self, idx, mask):
        mask_attend = gather_nodes(mask.unsqueeze(-1), idx).squeeze(-1)
        mask_attend = mask.unsqueeze(-1) * mask_attend
        return mask_attend

    def _get_decoder_mask(self, idx, mask):
        mask_attend = self._autoregressive_mask(idx).unsqueeze(-1)
        mask_1D = mask.view([mask.size(0), mask.size(1), 1, 1])
        mask_bw = mask_1D * mask_attend
        mask_fw = mask_1D * (1. - mask_attend)
        return mask_bw, mask_fw

    def _full_dist(self, X, mask, top_k=30, eps=1E-6):
        mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
        dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
        D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

        D_max, _ = torch.max(D, -1, keepdim=True)
        D_adjust = D + (1. - mask_2D) * D_max
        D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
        return D_neighbors, E_idx  

    def _encoder_network(self, h_V, h_P, h_F, P_idx, F_idx, mask):
        P_idx_mask_attend = self._get_encoder_mask(P_idx, mask) # part
        F_idx_mask_attend = self._get_encoder_mask(F_idx, mask) # full
        for (local_layer, global_layer) in self.encoder_layers:
            # local_layer
            h_EV_local = cat_neighbors_nodes(h_V, h_P, P_idx)
            h_V = local_layer(h_V, h_EV_local, mask_V=mask, mask_attend=P_idx_mask_attend)
            # global layer
            h_EV_global = cat_neighbors_nodes(h_V, h_F, F_idx)
            h_V = h_V + global_layer(h_V, h_EV_global, mask_V=mask, mask_attend=F_idx_mask_attend)
        return h_V

    def _get_sv_encoder(self, S, h_V, h_P, P_idx):
        h_S = self.W_s(S)
        h_PS = cat_neighbors_nodes(h_S, h_P, P_idx)
        h_PS_encoder = cat_neighbors_nodes(torch.zeros_like(h_S), h_P, P_idx) 
        h_PSV_encoder = cat_neighbors_nodes(h_V, h_PS_encoder, P_idx)
        return h_PS, h_PSV_encoder

    def _get_features(self, X, mask):
        X_ca = X[:,:,1,:]
        D_neighbors, F_idx = self._full_dist(X_ca, mask, 500)
        P_idx = F_idx[:, :, :self.top_k].clone()

        _V = _dihedrals(X)
        _V = self.norm_nodes(self.node_embedding(_V))
        _F = torch.cat((_rbf(D_neighbors, self.num_rbf), _orientations_coarse_gl(X_ca, F_idx)), -1)
        _F = self.norm_edges(self.edge_embedding(_F))
        _P = _F[..., :self.top_k, :]
    
        h_V = self.W_v(_V)
        h_P, h_F = self.W_e(_P), self.W_f(_F)
        return h_V, h_P, h_F, P_idx, F_idx

    def forward(self, X, S, mask, **kwargs):
        h_V, h_P, h_F, P_idx, F_idx = self._get_features(X=X, mask=mask)
        h_V = self._encoder_network(h_V, h_P, h_F, P_idx, F_idx, mask)
        h_PS, h_PSV_encoder = self._get_sv_encoder(S, h_V, h_P, P_idx)
        # Decoder
        P_idx_mask_bw, P_idx_mask_fw = self._get_decoder_mask(P_idx, mask)
        for local_layer in self.decoder_layers:
            # local_layer
            h_ESV_local = cat_neighbors_nodes(h_V, h_PS, P_idx)
            h_ESV_local = P_idx_mask_bw * h_ESV_local + P_idx_mask_fw * h_PSV_encoder
            h_V = local_layer(h_V, h_ESV_local, mask_V=mask)
        logits = self.W_out(h_V) 
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs

    def sample(self, X, mask=None, temperature=0.1, **kwargs):
        h_V, h_P, h_F, P_idx, F_idx = self._get_features(X=X, mask=mask)
        h_V = self._encoder_network(h_V, h_P, h_F, P_idx, F_idx, mask)
        # Decoder
        P_idx_mask_bw, P_idx_mask_fw = self._get_decoder_mask(P_idx, mask)
        N_batch, N_nodes = X.size(0), X.size(1)
        h_S = torch.zeros_like(h_V)
        S = torch.zeros((N_batch, N_nodes), dtype=torch.int64, device=X.device)
        h_V_stack = [h_V] + [torch.zeros_like(h_V) for _ in range(len(self.decoder_layers))]
        for t in range(N_nodes):
            # Hidden layers
            P_idx_t = P_idx[:,t:t+1,:]
            h_P_t = h_P[:,t:t+1,:,:]
            h_PS_t = cat_neighbors_nodes(h_S, h_P_t, P_idx_t)
            h_PSV_encoder_t = P_idx_mask_fw[:,t:t+1,:,:] * cat_neighbors_nodes(h_V, h_PS_t, P_idx_t)
            for l, local_layer in enumerate(self.decoder_layers):
                # local layer
                h_PSV_decoder_t = cat_neighbors_nodes(h_V_stack[l], h_PS_t, P_idx_t)
                h_V_t = h_V_stack[l][:,t:t+1,:]
                h_PSV_t = P_idx_mask_bw[:,t:t+1,:,:] * h_PSV_decoder_t + h_PSV_encoder_t
                h_V_stack[l+1][:,t,:] = local_layer(
                    h_V_t, h_PSV_t, mask_V=mask[:, t:t+1]
                ).squeeze(1)
            # Sampling step
            h_V_t = h_V_stack[-1][:,t,:]
            logits = self.W_out(h_V_t) / temperature
            probs = F.softmax(logits, dim=-1)
            S_t = torch.multinomial(probs, 1).squeeze(-1)
            # Update
            h_S[:,t,:] = self.W_s(S_t)
            S[:,t] = S_t
        return S