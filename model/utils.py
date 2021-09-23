import torch
import torch.nn.functional as F

# Thanks for StructTrans
# https://github.com/jingraham/neurips19-graph-protein-design
def nan_to_num(tensor, nan=0.0):
    idx = torch.isnan(tensor)
    tensor[idx] = nan
    return tensor

def _normalize(tensor, dim=-1):
    return nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

def _dihedrals(X, eps=1e-7):
    X = X[:,:,:3,:].reshape(X.shape[0], 3*X.shape[1], 3)

    dX = X[:,1:,:] - X[:,:-1,:]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:,:-2,:]
    u_1 = U[:,1:-1,:]
    u_0 = U[:,2:,:]

    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

    D = F.pad(D, (1,2), 'constant', 0)
    D = D.view((D.size(0), int(D.size(1)/3), 3))
    phi, psi, omega = torch.unbind(D,-1)

    D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
    return D_features

def _hbonds(X, E_idx, mask_neighbors, eps=1E-3):
    X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

    X_atoms['C_prev'] = F.pad(X_atoms['C'][:,1:,:], (0,0,0,1), 'constant', 0)
    X_atoms['H'] = X_atoms['N'] + F.normalize(
            F.normalize(X_atoms['N'] - X_atoms['C_prev'], -1)
        +  F.normalize(X_atoms['N'] - X_atoms['CA'], -1)
    , -1)

    def _distance(X_a, X_b):
        return torch.norm(X_a[:,None,:,:] - X_b[:,:,None,:], dim=-1)

    def _inv_distance(X_a, X_b):
        return 1. / (_distance(X_a, X_b) + eps)

    U = (0.084 * 332) * (
            _inv_distance(X_atoms['O'], X_atoms['N'])
        + _inv_distance(X_atoms['C'], X_atoms['H'])
        - _inv_distance(X_atoms['O'], X_atoms['H'])
        - _inv_distance(X_atoms['C'], X_atoms['N'])
    )

    HB = (U < -0.5).type(torch.float32)
    neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1),  E_idx)
    return neighbor_HB

def _rbf(D, num_rbf):
    D_min, D_max, D_count = 0., 20., num_rbf
    D_mu = torch.linspace(D_min, D_max, D_count).to(D.device)
    D_mu = D_mu.view([1,1,1,-1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)
    RBF = torch.exp(-((D_expand - D_mu) / D_sigma)**2)
    return RBF

def _orientations_coarse(X, E_idx, eps=1e-6):
    dX = X[:,1:,:] - X[:,:-1,:]
    U = F.normalize(dX, dim=-1)
    u_2 = U[:,:-2,:]
    u_1 = U[:,1:-1,:]
    u_0 = U[:,2:,:]

    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

    cosA = -(u_1 * u_0).sum(-1)
    cosA = torch.clamp(cosA, -1+eps, 1-eps)
    A = torch.acos(cosA)

    cosD = (n_2 * n_1).sum(-1)
    cosD = torch.clamp(cosD, -1+eps, 1-eps)
    D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

    AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
    AD_features = F.pad(AD_features, (0,0,1,2), 'constant', 0)

    o_1 = F.normalize(u_2 - u_1, dim=-1)
    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
    O = O.view(list(O.shape[:2]) + [9])
    O = F.pad(O, (0,0,1,2), 'constant', 0)

    O_neighbors = gather_nodes(O, E_idx)
    X_neighbors = gather_nodes(X, E_idx)
    
    O = O.view(list(O.shape[:2]) + [3,3])
    O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

    dX = X_neighbors - X.unsqueeze(-2)
    dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
    dU = F.normalize(dU, dim=-1)
    R = torch.matmul(O.unsqueeze(2).transpose(-1,-2), O_neighbors)
    Q = _quaternions(R)

    O_features = torch.cat((dU,Q), dim=-1)
    return AD_features, O_features

def _orientations_coarse_gl(X, E_idx, eps=1e-6):
    dX = X[:,1:,:] - X[:,:-1,:]
    U = F.normalize(dX, dim=-1)
    u_2, u_1 = U[:,:-2,:], U[:,1:-1,:]
    n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
    o_1 = F.normalize(u_2 - u_1, dim=-1)

    O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
    O = O.view(list(O.shape[:2]) + [9])
    O = F.pad(O, (0,0,1,2), 'constant', 0)

    O_neighbors = gather_nodes(O, E_idx)
    X_neighbors = gather_nodes(X, E_idx)

    O = O.view(list(O.shape[:2]) + [3,3]).unsqueeze(2)
    O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3,3])

    dX = X_neighbors - X.unsqueeze(-2)
    dU = torch.matmul(O, dX.unsqueeze(-1)).squeeze(-1)
    R = torch.matmul(O.transpose(-1,-2), O_neighbors)
    return torch.cat((F.normalize(dU, dim=-1), _quaternions(R)), dim=-1)

def _contacts(D_neighbors, mask_neighbors, cutoff=8):
    D_neighbors = D_neighbors.unsqueeze(-1)
    return mask_neighbors * (D_neighbors < cutoff).type(torch.float32)

def _dist(X, mask, top_k=30, eps=1E-6):
    mask_2D = torch.unsqueeze(mask,1) * torch.unsqueeze(mask,2)
    dX = torch.unsqueeze(X,1) - torch.unsqueeze(X,2)
    D = mask_2D * torch.sqrt(torch.sum(dX**2, 3) + eps)

    D_max, _ = torch.max(D, -1, keepdim=True)
    D_adjust = D + (1. - mask_2D) * D_max
    D_neighbors, E_idx = torch.topk(D_adjust, min(top_k, D_adjust.shape[-1]), dim=-1, largest=False)
    mask_neighbors = gather_edges(mask_2D.unsqueeze(-1), E_idx)
    return D_neighbors, E_idx, mask_neighbors    

def gather_edges(edges, neighbor_idx):
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    return torch.gather(edges, 2, neighbors)

def gather_nodes(nodes, neighbor_idx):
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    return neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])

def gather_nodes_t(nodes, neighbor_idx):
    idx_flat = neighbor_idx.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    return torch.gather(nodes, 1, idx_flat)

def cat_neighbors_nodes(h_nodes, h_neighbors, E_idx):
    h_nodes = gather_nodes(h_nodes, E_idx)
    return torch.cat([h_neighbors, h_nodes], -1)

def _quaternions(R):
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    Rxx, Ryy, Rzz = diag.unbind(-1)
    magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz, 
        - Rxx + Ryy - Rzz, 
        - Rxx - Ryy + Rzz
    ], -1)))
    _R = lambda i,j: R[:,:,:,i,j]
    signs = torch.sign(torch.stack([
        _R(2,1) - _R(1,2),
        _R(0,2) - _R(2,0),
        _R(1,0) - _R(0,1)
    ], -1))
    xyz = signs * magnitudes
    w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.
    Q = torch.cat((xyz, w), -1)
    return F.normalize(Q, dim=-1)