import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_dense_batch


def sinusoidal_embedding(n, d):
    """
    Returns the sinusoidal positional encoding of dimension d for n timesteps.
    """
    embedding = torch.tensor([[i / 10_000 ** (2 * j / d) for j in range(d)] for i in range(n)])
    sin_mask = torch.arange(0, n, 2)

    embedding[sin_mask] = torch.sin(embedding[sin_mask])
    embedding[1 - sin_mask] = torch.cos(embedding[sin_mask])

    return embedding

class AddEdgeStatePE(BaseTransform):
    """
    A transform class to add data about adjacent edges state in each nodes' embedding

    This implementation is not efficient at all it is just meant for little tests, which aren't conclusive.
    """
    def __init__(self) -> None:
        pass

    def forward(self, data) -> Data:
        assert data.edge_index is not None
        assert data.edge_state is not None
        assert data.batch is not None

        adjacent_edges = to_dense_batch(data.edge_state, data.edge_index[0])[0] #For each node, the states of its adjacent edges (in the underlying graph)
        adjacent_edges = torch.flatten(adjacent_edges, start_dim=1)

        data.x = torch.hstack((data.x, adjacent_edges))

        return data
    
class D3PM(nn.Module):
    """
    A class to represent a diffusion model using the network parameter as a denoiser.
    """
    def __init__(self, network, device, num_hops=3, time_emb_dim=100, num_timesteps=1000):
        super(D3PM, self).__init__()
        self.num_hops = num_hops
        self.edges_transform = AddEdgeStatePE()
        self.network = network
        self.device = device
        self.num_nodes = 4 * (2**num_hops)
        self.num_timesteps = num_timesteps
        self.time_emb_dim = time_emb_dim
        self.eps = torch.linspace(0, 0.5, self.num_timesteps).to(device)
        self.eps = torch.cat((self.eps, torch.tensor([0.5]).to(device)))
        self.time_emb = sinusoidal_embedding(self.num_timesteps+1, self.time_emb_dim).to(device)

    def add_noise(self, G_start, thresholds, t):

        noised_G = G_start.clone()

        noised_indices = (noised_G.x[noised_G.edge_index[0]][:,0] < thresholds[noised_G.batch[noised_G.edge_index[0]]]) | (noised_G.x[noised_G.edge_index[1]][:,0] < thresholds[noised_G.batch[noised_G.edge_index[1]]])
        epsilon = self.eps[t[noised_G.batch[noised_G.edge_index[0]]] + noised_indices.long()]
        
        p = (1 - epsilon) * noised_G.edge_state[:, 0] + epsilon * noised_G.edge_state[:, 1]
        sample = torch.bernoulli(p).long().to(self.device)
        
        noised_G.edge_state = F.one_hot(1 - sample, num_classes=2).to(torch.float32)
        noised_G.edge_weight = noised_G.edge_state[:,0]
        
        return noised_G

    def reverse(self, batch, thresholds, t):
        batch.x = F.one_hot((batch.x[:,0] < thresholds[batch.batch]).long(), num_classes=2).to(torch.float32)
        batch = self.edges_transform(batch)

        batch.x = torch.cat([batch.x, self.time_emb[t[batch.batch] + batch.x[:,1].long()]], dim=1)

        pos = (torch.arange(batch.num_nodes, device=self.device) % batch.ptr[1]) % (2*self.num_hops)
        
        batch.x = torch.cat([batch.x, F.one_hot( pos, num_classes=(2*self.num_hops)).to(torch.float32)], dim=1)

        res = self.network(x=batch.x, edge_index=batch.edge_index)

        return res
    
class D3PM_variant(nn.Module):
    """
    A class to represent a diffusion model using the network parameter as a denoiser.
    """
    def __init__(self, network, device, num_hops=3, time_emb_dim=100, num_timesteps=1000):
        super(D3PM_variant, self).__init__()
        self.num_hops = num_hops
        self.edges_transform = AddEdgeStatePE()
        self.network = network
        self.device = device
        self.num_nodes = 4 * (2**num_hops)
        self.num_timesteps = num_timesteps
        self.time_emb_dim = time_emb_dim
        self.eps = torch.linspace(0, 0.5, self.num_timesteps).to(device)
        self.time_emb = sinusoidal_embedding(self.num_timesteps, self.time_emb_dim).to(device)

    def add_noise(self, G_start, thresholds, t):

        noised_G = G_start.clone()
        
        epsilon = self.eps[t[noised_G.batch[noised_G.edge_index[0]]]]
        noised_indices = (noised_G.x[noised_G.edge_index[0]][:,0] < thresholds[noised_G.batch[noised_G.edge_index[0]]]) | (noised_G.x[noised_G.edge_index[1]][:,0] < thresholds[noised_G.batch[noised_G.edge_index[1]]])
        
        p = (1 - epsilon) * noised_G.edge_state[:, 0] + epsilon * noised_G.edge_state[:, 1]
        sample = torch.bernoulli(p).long().to(self.device)
        
        noised_G.edge_state[noised_indices] = F.one_hot(1 - sample, num_classes=2).to(torch.float32)[noised_indices]
        noised_G.edge_weight = noised_G.edge_state[:,0]
        
        return noised_G

    def reverse(self, batch, thresholds, t):
        batch.x = F.one_hot((batch.x[:,0] < thresholds[batch.batch]).long(), num_classes=2).to(torch.float32)
        batch = self.edges_transform(batch)

        batch.x = torch.cat([batch.x, self.time_emb[t[batch.batch]]*batch.x[:,1].unsqueeze(1)], dim=1)

        pos = (torch.arange(batch.num_nodes, device=self.device) % batch.ptr[1]) % (2*self.num_hops)
        
        batch.x = torch.cat([batch.x, F.one_hot( pos, num_classes=(2*self.num_hops)).to(torch.float32)], dim=1)

        res = self.network(x=batch.x, edge_index=batch.edge_index)

        return res
    
class Network(nn.Module):

    """
    A basic GNN model to denoise in the diffusion process.
    """
    def __init__(self, max_deg=4, num_gcn_layers=3, num_mlp1_layers=2, num_mlp2_layers=2, time_emb_dim=100):
        super().__init__()
        self.feature_size = 2 + 2*max_deg + 2*num_gcn_layers + time_emb_dim
        self.mlp1 = MLP(in_channels=self.feature_size, hidden_channels=[2*self.feature_size for _ in range(num_mlp1_layers)])
        
        self.GNN = torch_geometric.nn.GCN(in_channels=2*self.feature_size, 
                                          hidden_channels=4*self.feature_size, 
                                          num_layers=num_gcn_layers, 
                                          out_channels=4*self.feature_size,
                                         )

        self.mlp2 = MLP(in_channels=8*self.feature_size, hidden_channels=[8*self.feature_size for _ in range(num_mlp2_layers)]+[1])
        self.sig = torch.nn.Sigmoid()

        self.edges_transform = AddEdgeStatePE()
    
    def forward(self, x, edge_index):

        y = self.mlp1(x)
        y = self.GNN(x=y, edge_index=edge_index)
        y = torch.cat((y[edge_index[0]].T, y[edge_index[1]].T)).T
        out = self.mlp2(y)

        return self.sig(out).reshape(-1)
    