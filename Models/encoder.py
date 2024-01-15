import torch
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims
from torch import cat, zeros_like

from Models.utils import device


class NodeEncoder(torch.nn.Module):

    def __init__(self, emb_dim, feature_dims=None, uses_k_wl_transform=0, k_wl_separate=False):
        super(NodeEncoder, self).__init__()
        self.atom_embedding_list = torch.nn.ModuleList()
        if uses_k_wl_transform > 0 and k_wl_separate:
            emb_dim_local = emb_dim // 2
            self.k_wl_separate = True
        else:
            self.k_wl_separate = False
            emb_dim_local = emb_dim
        if feature_dims is None:
            feature_dims = get_atom_feature_dims()
        self.feature_dims = feature_dims
        if uses_k_wl_transform > 0:
            self.uses_k_wl_transform = bool(uses_k_wl_transform)
            self.k_wl_embeddings = []
            for i in range(uses_k_wl_transform + 1):
                emb = torch.nn.Embedding(10, emb_dim_local)
                torch.nn.init.xavier_uniform_(emb.weight.data)
                emb.to(device())
                self.k_wl_embeddings.append(emb)
        print('node embedding feature dims', feature_dims)
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim_local)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            emb.to(device())
            self.atom_embedding_list.append(emb)
        self.len_embedding_list = len(self.atom_embedding_list)

    def forward(self, x, k_wl=0):
        x_embedding = 0
        x = x.long()
        k_wl_embedding = None
        try:
            for i in range(x.shape[1]):
                if k_wl > 0 and i == 0 and self.uses_k_wl_transform:
                    k_wl_embedding = self.k_wl_embeddings[k_wl](x[:, i])
                else:
                    if i >= self.len_embedding_list:
                        # the first position is not repeating
                        x_embedding += self.atom_embedding_list[(i - 1) % self.len_embedding_list](x[:, i])
                    else:
                        x_embedding += self.atom_embedding_list[i - 1 if k_wl > 0 else i](x[:, i])
            if self.k_wl_separate:
                if not isinstance(x_embedding, int):
                    return cat((k_wl_embedding, x_embedding), dim=1)
                else:
                    return cat((k_wl_embedding, zeros_like(k_wl_embedding)), dim=1)
            else:
                if k_wl_embedding is not None:
                    x_embedding += k_wl_embedding
                return x_embedding
        except Exception as e:
            print('feature dims', self.feature_dims)
            print('len feature dims', len(self.feature_dims))
            print('i', i)
            print('x shape', x.shape)
            print('k', k_wl)
            print(x[:, i])
            raise e


class EdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim, feature_dims=None, uses_k_wl_transform=False, k_wl_separate=False):
        super(EdgeEncoder, self).__init__()
        self.bond_embedding_list = torch.nn.ModuleList()
        if uses_k_wl_transform and k_wl_separate:
            emb_dim_local = emb_dim // 2
            self.k_wl_separate = True
        else:
            self.k_wl_separate = False
            emb_dim_local = emb_dim
        if feature_dims is None:
            feature_dims = get_bond_feature_dims()

        if uses_k_wl_transform:
            feature_dims = [100] + feature_dims
        for i, dim in enumerate(feature_dims):
            emb = torch.nn.Embedding(dim, emb_dim_local)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)
        print('bond_embedding_list', self.bond_embedding_list)
        self.len_embedding_list = len(self.bond_embedding_list)

    def forward(self, edge_attr):
        bond_embedding = 0
        for i in range(edge_attr.shape[1]):
            if i == 0 and self.k_wl_separate:
                k_wl_embedding = self.bond_embedding_list[i](edge_attr[:, i])
            else:
                if i >= self.len_embedding_list:
                    # the first position is not repeating
                    bond_embedding += self.bond_embedding_list[(i - 1) % (self.len_embedding_list - 1) + 1](
                        edge_attr[:, i])
                else:
                    bond_embedding += self.bond_embedding_list[i](edge_attr[:, i])
        if self.k_wl_separate:
            if not isinstance(bond_embedding, int):
                return cat((k_wl_embedding, bond_embedding), dim=1)
            else:
                return cat((k_wl_embedding, zeros_like(k_wl_embedding)), dim=1)
        else:
            return bond_embedding


class EgoEncoder(torch.nn.Module):
    # From ESAN
    def __init__(self, encoder):
        super(EgoEncoder, self).__init__()
        self.num_added = 2
        self.enc = encoder

    def forward(self, x):
        return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:])))


class ZincAtomEncoder(torch.nn.Module):
    # From ESAN
    def __init__(self, policy, emb_dim):
        super(ZincAtomEncoder, self).__init__()
        self.policy = policy
        self.num_added = 2
        self.enc = torch.nn.Embedding(21, emb_dim)

    def forward(self, x):
        if self.policy == 'ego_nets_plus':
            return torch.hstack((x[:, :self.num_added], self.enc(x[:, self.num_added:].squeeze())))
        else:
            return self.enc(x.squeeze())


class KWlEmbeddings(torch.nn.Module):
    def __init__(self, k, emb_dim):
        super(KWlEmbeddings, self).__init__()
        self.emb = torch.nn.Embedding(k ** 2, emb_dim)
        self.emb.to(device())
        self.emb_dim = emb_dim
        torch.nn.init.xavier_uniform_(self.emb.weight.data)

    def forward(self, x):
        return self.emb(x)
