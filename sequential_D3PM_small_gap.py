import networkx as nx
import matplotlib.pyplot as plt
import torch
from torch import nn
import numpy as np
from torchvision.ops import MLP
import torch_geometric
from torch_geometric.data import Data, Dataset
from torch_geometric.transforms import BaseTransform
from torch.nn import functional as F
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import to_dense_batch
import random
from copy import deepcopy
import os
import os.path as osp
from tqdm import tqdm
from itertools import combinations
from sys import stdout

def wilson_algorithm(G, start):
    """
    Wilson's algorithm for generating a uniform spanning tree on a graph.

    Parameters:
    - G: NetworkX graph
    - start: starting node for the algorithm

    Returns:
    - T: Uniform spanning tree of G
    """
    U = {start}
    T = deepcopy(G)
    nx.set_edge_attributes(T, {e:{"edge_weight":0.} for e in T.edges(data=False)})
    nx.set_node_attributes(T, {n:{'pos':n} for n in T.nodes(data=False)})
    while len(U) < len(G.nodes):
        u = random.choice(list(set(G.nodes) - U))
        path = [u]
        while u not in U:
            u = random.choice(list(G.neighbors(u)))
            if u in path:
                cycle_index = path.index(u)
                path = path[:cycle_index + 1]
            else:
                path.append(u)
        U.update(path)
        nx.set_edge_attributes(T, {(path[i], path[i + 1]):{"edge_weight":1.} for i in range(len(path) - 1)})
    return T

def generate_ust_maze(width, height):
    """
    Generates a maze using Wilson's algorithm.

    Parameters:
    - width: width of the maze
    - height: height of the maze

    Returns:
    - T: Uniform spanning tree representing the maze
    """
    G = nx.grid_2d_graph(width, height)
    start = (random.randint(0, width-1), random.randint(0, height-1))
    T = wilson_algorithm(G, start)
    return T

def generate_ust_maze_list(size, width, height):
    """
    Generates a maze dataset

    Parameters:
    - width: width of the maze
    - height: height of the maze

    Returns:
    - L a list of maze
    """
    L = []
    for i in range(size):
        L.append(generate_ust_maze(width, height))
    
    return L

def write_ust_maze_list(L, dir_name):
    """
    Writes a list of mazes in a specified file

    Parameters:
    - L: list of ust mazes
    - dir_name: The name of the directory to write the list
    """

    for i, G in enumerate(L):
        nx.write_gml(G, dir_name+"/Graph_"+str(i), stringizer=nx.readwrite.gml.literal_stringizer)

def draw_maze(T, width, height, title):
    """
    Draw the maze represented by the uniform spanning tree T.

    Parameters:
    - T: Uniform spanning tree representing the maze
    - width: width of the maze
    - height: height of the maze
    """
    pos = {n:pos for (n,pos) in nx.get_node_attributes(T, "pos").items()}
    real_edges = [e for (e,v) in nx.get_edge_attributes(T, "edge_weight").items() if v==1]
    plt.figure(figsize=(10, 10))
    nx.draw(T.edge_subgraph(real_edges), pos=pos, with_labels=False, node_size=10, width=2, edge_color='blue')
    plt.xlim(-1, width)
    plt.ylim(-1, height)
    plt.gca().invert_yaxis()
    plt.savefig(title)

def draw_Data(G, width, height, title):
    """
    Draw ther maze represented by the graph G

    Parameters:
    - G: torch_geometric graph
    - width: width of the maze
    - height: height of the maze
    """

    T = to_networkx(G, to_undirected=True, node_attrs=['pos'], edge_attrs=['edge_weight'])
    draw_maze(T, width, height, title)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using gpu: %s ' % torch.cuda.is_available())

class Maze_dataset(Dataset):

    """
    A class to represent a Dataset of maze
    """
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, force_reload=False):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.pre_transform = pre_transform
        self.transform = transform
        self.pre_filter = pre_filter
        self.root = root

    @property
    def raw_file_names(self):
        return [f for f in os.listdir(self.root+"/raw")]

    @property
    def processed_file_names(self):
        return [f for f in os.listdir(self.root+"/processed")]

    def process(self):
        idx = 0
        for raw_path in self.raw_paths:
            # Read data from `raw_path`.
            raw_graph = nx.read_gml(raw_path)

            data = from_networkx(raw_graph).to(device)

            data.edge_index, data.edge_weight = torch_geometric.utils.coalesce(edge_index=data.edge_index, edge_attr=data.edge_weight)

            data.edge_state = torch.hstack((data.edge_weight.unsqueeze(1), 1. - data.edge_weight.unsqueeze(1))) #Binary state space

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'data_{idx}.pt'))
        return data
    
    def get_raw(self, idx):
        G = nx.read_gml(osp.join(self.raw_dir, f'Graph_{idx}'), destringizer=nx.readwrite.gml.literal_destringizer)
        return G
    
#L = generate_ust_maze_list(10_000, 9, 9)
#write_ust_maze_list(L, "D3PM/Ddataset9/raw")

dataset = Maze_dataset("Ddataset32")
width, height = 32, 32

print("Loaded dataset")

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
    def __init__(self, network, num_hops=3, time_emb_dim=100, num_timesteps=1000, device=device):
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
        sample = torch.bernoulli(p).long().to(device)
        
        noised_G.edge_state = F.one_hot(1 - sample, num_classes=2).to(torch.float32)

        undirected_indices = (noised_G.edge_index[0] < noised_G.edge_index[1]) * noised_G.csr()[2] + (noised_G.edge_index[0] >= noised_G.edge_index[1]) * noised_G.csc()[2] # To noise in a undirected way. 
        noised_G.edge_state[undirected_indices] = noised_G.edge_state[undirected_indices]
        
        noised_G.edge_weight = noised_G.edge_state[:,0]
        
        return noised_G

    def reverse(self, batch, thresholds, t):
        batch.x = F.one_hot((batch.x[:,0] < thresholds[batch.batch]).long(), num_classes=2).to(torch.float32)
        batch = self.edges_transform(batch)

        batch.x = torch.cat([batch.x, self.time_emb[t[batch.batch] + batch.x[:,1].long()]], dim=1)

        pos = (torch.arange(batch.num_nodes, device=device) % batch.ptr[1]) % (2*self.num_hops)
        
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
    
print("Starting training")

num_timesteps = 10
time_emb_dim = 100
batch_size = 2
data_size = 10000
num_gcn_layers = 4
num_mlp1_layers = 3
num_mlp2_layers = 3

graph_size = width * height

network = Network(max_deg=4, num_gcn_layers=num_gcn_layers, num_mlp1_layers=num_mlp1_layers, num_mlp2_layers=num_mlp2_layers, time_emb_dim=time_emb_dim)

model = D3PM(network, num_hops=num_gcn_layers, time_emb_dim=time_emb_dim, num_timesteps=num_timesteps, device=device)
#model.load_state_dict(torch.load('model_weights2.pth', weights_only=True))

train_loader = torch_geometric.loader.DataLoader(dataset[:data_size], batch_size=batch_size, shuffle=True)

num_epochs = 100
learning_rate = 0.001

optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
loss_fn = torch.nn.BCELoss()

losses = []
accs = []

print(device)
model.to(device)
for epoch in range(num_epochs):
        stdout.write(f"Epoch {epoch}")
        running_loss = 0.0
        running_acc = 0.0
        model.train()

        # inside the training loop
        for step, batch in enumerate(train_loader):

                batch = batch.to(device)

                num_nodes = batch.ptr[1:] - batch.ptr[:-1]
                
                batch.x = torch.arange(batch.num_nodes).to(device).reshape(-1, 1).float() - batch.ptr[batch.batch].reshape(-1, 1).float()
                thresholds = torch.randint(low=1, high=graph_size, size=(batch.num_graphs,)).to(device)
                t = torch.randint(low=0, high=num_timesteps, size=(batch.num_graphs,)).to(device).long()

                """
                Noising
                """

                noisy = model.add_noise(batch, thresholds, t)
                
                """
                Denoising
                """

                noise_pred = model.reverse(batch=noisy, thresholds=thresholds, t=t)
                
                loss = loss_fn(noise_pred, batch.edge_state[:,0])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item() / len(train_loader)
                running_acc += torch.mean(((noise_pred > 0.5) == batch.edge_state[:,0]).float()).item() / len(train_loader)
                
        losses.append(running_loss)
        accs.append(running_acc)
        scheduler.step(running_loss)
        print("\tRunning loss: "+str(running_loss))
        print("\tRunning acc: "+str(running_acc))
        print("-")
        torch.save(model.state_dict(), 'model_weights2.pth')

fig, ax = plt.subplots(2, 1, figsize=(20,10))
ax[0].plot(losses)
ax[0].set_title("BCELoss")
ax[1].plot(accs)
ax[1].set_title("Accuracy")
plt.savefig("prog_sg.jpg")

print("Training done")
print("Starting generation")

def num_squares(G):
    res = 0
    for v in range(G.number_of_nodes()):
        for u, w in combinations(G[v], 2):
            if u>v and w>v:
                squares = len((set(G[u]) & set(G[w])) - {v})
                res += squares
    return res

def cc_and_trees(G):
    cc = nx.connected_components(G)
    cc = list([c for c in cc if len(c)>1])
    num_cc = len(cc)
    tree_ratio = np.sum([nx.is_tree(G.subgraph(c))for c in cc])/num_cc

    return num_cc, tree_ratio

def analytics(data):
    data2 = deepcopy(data)
    data2.edge_index = torch.tensor([[data2.edge_index[0][i], data2.edge_index[1][i]] for i in range(data2.num_edges) if data2.edge_weight[i] == 1]).T
    G = to_networkx(data2, to_undirected=True, node_attrs=['pos'], edge_attrs=['edge_weight'])

    return *cc_and_trees(G), num_squares(G)

G = nx.grid_2d_graph(width, height)
nx.set_edge_attributes(G, {e:{"edge_weight":1.} for e in G.edges(data=False)})
nx.set_node_attributes(G, {n:{'pos':n} for n in G.nodes(data=False)})
data = from_networkx(G).to(device)
data.batch = torch.zeros(data.num_nodes).long()
data.edge_index, data.edge_weight = torch_geometric.utils.coalesce(edge_index=data.edge_index, edge_attr=data.edge_weight)
data.edge_state = torch.hstack((data.edge_weight.unsqueeze(1), 1. - data.edge_weight.unsqueeze(1)))

data = data.to(device)
model.to(device)
model.eval()
t = torch.tensor([num_timesteps-1]).to(device)
thresholds = torch.tensor([data.num_nodes - 1]).to(device)
data.x = torch.arange(data.num_nodes).to(device).reshape(-1, 1).float()
noise = model.add_noise(G_start=data, thresholds=thresholds, t=t)

print("Initial graph")
cc, tree_ratio, squares = analytics(data)
print("Number of connected component: "+str(cc))
print("Ratio of trees: "+str(tree_ratio))
print("Number of squares: "+str(squares))
draw_Data(data, width, height, "initial_sg.jpg")

print("Noise")
cc, tree_ratio, squares = analytics(noise)
print("Number of connected component: "+str(cc))
print("Ratio of trees: "+str(tree_ratio))
print("Number of squares: "+str(squares))
draw_Data(noise, width, height, "noise_sg.jpg")

noise1 = deepcopy(noise)
noise1.ptr = torch.tensor([0, noise1.num_nodes]).to(device)
noise1.batch = torch.zeros(noise1.num_nodes).long().to(device)

thresholds = torch.arange(noise1.num_nodes).flip(0).unsqueeze(1).to(device)
timesteps = torch.arange(num_timesteps).flip(0).unsqueeze(1).to(device)

for t in timesteps[:-1]:
    for threshold in thresholds:
        pred = model.reverse(noise1, threshold, t)

        sample = torch.bernoulli(pred).long()
        noise1.edge_state = F.one_hot( 1 - sample, num_classes=2).float()

        undirected_indices = (noise1.edge_index[0] < noise1.edge_index[1]) * noise1.csr()[2] + (noise1.edge_index[0] >= noise1.edge_index[1]) * noise1.csc()[2] 
        noise1.edge_state = noise1.edge_state[undirected_indices]

        noise1.edge_weight = noise1.edge_state[:,0]

print("Generated tree")
cc, tree_ratio, squares = analytics(noise1)
print("Number of connected component: "+str(cc))
print("Ratio of trees: "+str(tree_ratio))
print("Number of squares: "+str(squares))
draw_Data(noise1, width, height, "generated_sg.jpg")