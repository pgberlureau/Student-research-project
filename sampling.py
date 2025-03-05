import networkx as nx
import os
import os.path as osp
import torch
import numpy as np
import torch_geometric
from torch.nn import functional as F
from torch_geometric.utils.convert import to_networkx, from_networkx
from copy import deepcopy
from itertools import combinations
from dataset import draw_Data
from torch_geometric.loader import DataLoader
import argparse
from pathlib import Path
from tqdm import tqdm

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

def generate_noise(width, height, model, device, num_timesteps):

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

    return noise

def sample_from_noise(noise, width, height, device, num_timesteps, model, variant, noise_stats=True, generated_stats=True, noise_png_name=None, generated_png_name="generated.png"):

    if noise_stats:
        noise_cc, noise_tree_ratio, noise_squares = analytics(noise)
    else:
        noise_cc, noise_tree_ratio, noise_squares = None, None, None
    
    if noise_png_name is not None:
        draw_Data(noise, width, height, noise_png_name)

    noise.ptr = torch.tensor([0, noise.num_nodes]).to(device)
    noise.batch = torch.zeros(noise.num_nodes).long().to(device)

    thresholds = torch.arange(noise.num_nodes).flip(0).unsqueeze(1).to(device)
    timesteps = torch.arange(num_timesteps).flip(0).unsqueeze(1).to(device)

    if variant:
        for i, threshold in enumerate(thresholds):
            for idx, t in enumerate(timesteps):
                pred = model.reverse(noise, threshold, t)

                sample = torch.bernoulli(pred).long()
                noise.edge_state = F.one_hot( 1 - sample, num_classes=2).float()

                undirected_indices = (noise.edge_index[0] < noise.edge_index[1]) * noise.csr()[2] + (noise.edge_index[0] >= noise.edge_index[1]) * noise.csc()[2] 
                noise.edge_state = noise.edge_state[undirected_indices]

                noise.edge_weight = noise.edge_state[:,0]

                if idx+1 < len(timesteps):
                    noise = model.add_noise(noise, threshold, timesteps[idx+1])
                elif i+1 < len(thresholds):
                    noise = model.add_noise(noise, thresholds[i+1], timesteps[0])

    else:
        for i, t in enumerate(timesteps):
            for idx, threshold in enumerate(thresholds):
                pred = model.reverse(noise, threshold, t)

                sample = torch.bernoulli(pred).long()
                noise.edge_state = F.one_hot( 1 - sample, num_classes=2).float()

                undirected_indices = (noise.edge_index[0] < noise.edge_index[1]) * noise.csr()[2] + (noise.edge_index[0] >= noise.edge_index[1]) * noise.csc()[2] 
                noise.edge_state = noise.edge_state[undirected_indices]

                noise.edge_weight = noise.edge_state[:,0]

                if idx+1 < len(thresholds):
                    noise = model.add_noise(noise, thresholds[idx+1], t)
                elif i+1 < len(timesteps):
                    noise = model.add_noise(noise, thresholds[0], timesteps[i+1])

    if generated_stats:
        cc, tree_ratio, squares = analytics(noise)
    else:
        cc, tree_ratio, squares = None, None, None
    
    if generated_png_name is not None:
        draw_Data(noise, width, height, generated_png_name)

    return noise_cc, noise_tree_ratio, noise_squares, cc, tree_ratio, squares

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Sampling")

    parser.add_argument("--width", type=int, default=32, help="Width of the maze")
    parser.add_argument("--height", type=int, default=32, help="Height of the maze")
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model")
    parser.add_argument("--variant", action="store_true", required=False, help="Whether to use the variant method")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate")
    parser.add_argument("--samples_path", type=Path, required=False, help="Path to save the samples")
    parser.add_argument("--stats_path", type=Path, required=False, help="Compute statistics of the noise")

    args = parser.parse_args()
    args = vars(args)

    num_samples = args["num_samples"]
    width = args["width"]
    height = args["height"]
    model_path = args["model_path"]

    if "samples_path" in args:
        samples_path = args["samples_path"]
        args["samples_path"] = str(args["samples_path"])
        if not osp.exists(args["samples_path"]):
            os.makedirs(args["samples_path"])
        
    if "stats_path" in args:
        stats_parent = str(args["stats_path"].parent)
        if not osp.exists(stats_parent):
            os.makedirs(stats_parent)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    print("Loading model")
    model = torch.load(model_path, weights_only=False)

    print("Sampling")
    noise_cc_avg = 0
    noise_tree_ratio_avg = 0
    noise_squares_avg = 0
    cc_avg = 0
    tree_ratio_avg = 0
    squares_avg = 0

    for i in tqdm(range(num_samples)):
        if "samples_path" in args:
            generated_png_name = f"sample_{i}.png"
            figs_path = osp.join(str(args["samples_path"]), generated_png_name)
        else:
            generated_png_name = None
        
        noise = generate_noise(width, height, model, device, model.num_timesteps)
        noise_cc, noise_tree_ratio, noise_squares, cc, tree_ratio, squares = sample_from_noise(noise, width, height, device, model.num_timesteps, model, args["variant"], generated_png_name=figs_path)
        
        noise_cc_avg += noise_cc / num_samples
        noise_tree_ratio_avg += noise_tree_ratio / num_samples
        noise_squares_avg += noise_squares / num_samples
        cc_avg += cc / num_samples
        tree_ratio_avg += tree_ratio / num_samples
        squares_avg += squares / num_samples

    if "stats_path" in args:
        with open(args["stats_path"], 'w') as f:
            f.write(f"Average number of connected components in noise: {noise_cc_avg}\n")
            f.write(f"Average tree ratio in noise: {noise_tree_ratio_avg}\n")
            f.write(f"Average number of squares in noise: {noise_squares_avg}\n")
            f.write(f"Average number of connected components in generated: {cc_avg}\n")
            f.write(f"Average tree ratio in generated: {tree_ratio_avg}\n")
            f.write(f"Average number of squares in generated: {squares_avg}\n")

