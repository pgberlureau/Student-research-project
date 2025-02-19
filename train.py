import torch
from model import D3PM, D3PM_variant, Network
import argparse
from pathlib import Path
from dataset import Maze_dataset
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
import os
import os.path as osp
from tqdm import tqdm

def train_epoch(model_, loader, opt, loss_fn, dev, num_timesteps):
    running_loss = 0.0
    running_acc = 0.0

    # inside the training loop
    for batch in loader:
            batch = batch.to(dev)

            graph_size = batch.ptr[1]

            batch.x = torch.arange(batch.num_nodes).to(dev).reshape(-1, 1).float() - batch.ptr[batch.batch].reshape(-1, 1).float()
            thresholds = torch.randint(low=1, high=graph_size, size=(batch.num_graphs,)).to(dev)
            t = torch.randint(low=0, high=num_timesteps, size=(batch.num_graphs,)).to(dev).long()
            """
            Noising
            """
            noisy = model_.add_noise(batch, thresholds, t)
            
            """
            Denoising
            """
            noise_pred = model_.reverse(batch=noisy, thresholds=thresholds, t=t)
            
            loss = loss_fn(noise_pred, batch.edge_state[:,0])
            optimizer.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item() / len(loader)
            running_acc += torch.mean(((noise_pred > 0.5) == batch.edge_state[:,0]).float()).item() / len(loader)
    
    return running_loss, running_acc

def train_model(model_, loader, opt, loss_fn, num_epochs, sched, dev, num_timesteps):
    losses = []
    accuracies = []
    model_.to(dev)
    model_.train()
    for epoch in tqdm(range(num_epochs)):
            running_loss, running_acc = train_epoch(model_, loader, opt, loss_fn, dev, num_timesteps)
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {running_loss} - Accuracy: {running_acc}")
            losses.append(running_loss)
            accuracies.append(running_acc)
            sched.step(running_loss)
    
    return losses, accuracies

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--model_path", type=Path, required=True, help="Model's path")
    parser.add_argument("--dataset_path", type=Path, required=True, help="Dataset's path")
    parser.add_argument("--variant", action="store_true", required=False, help="Whether to use the variant method")
    parser.add_argument("--curves_path", type=Path, required=False, help="Path to the training curves")

    args = parser.parse_args()
    args = vars(args)

    num_timesteps = 10
    time_emb_dim = 100
    batch_size = 2
    data_size = 10000
    num_gcn_layers = 4
    num_mlp1_layers = 3
    num_mlp2_layers = 3

    model_path = str(args["model_path"])
    dataset_path = str(args["dataset_path"])
    curves_path = str(args["curves_path"])

    model_path_dir = str(Path(model_path).parent)
    if not osp.exists(model_path_dir):
        os.makedirs(model_path_dir)
    
    if "curves_path" in args:
        curves_path_dir = str(Path(curves_path).parent)
        if not osp.exists(curves_path_dir):
            os.makedirs(curves_path_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using gpu: %s ' % torch.cuda.is_available())

    dataset = Maze_dataset(dataset_path)

    print("Creating model")
    network = Network(max_deg=4, num_gcn_layers=num_gcn_layers, num_mlp1_layers=num_mlp1_layers, num_mlp2_layers=num_mlp2_layers, time_emb_dim=time_emb_dim)
    
    if args["variant"]:
        print("Using variant method")
        model = D3PM_variant(network, num_hops=num_gcn_layers, time_emb_dim=time_emb_dim, num_timesteps=num_timesteps, device=device)
    else:
        print("Using normal method")
        model = D3PM(network, num_hops=num_gcn_layers, time_emb_dim=time_emb_dim, num_timesteps=num_timesteps, device=device)

    train_loader = DataLoader(dataset[:data_size], batch_size=batch_size, shuffle=True)
    
    num_epochs = 100
    learning_rate = 0.001
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    loss_fn = torch.nn.BCELoss()

    print("Training model")
    losses, accuracies = train_model(model, train_loader, optimizer, loss_fn, num_epochs, scheduler, device, num_timesteps)
    print("Training finished")
    print(f"Loss: {losses[-1]}")
    print(f"Accuracy: {accuracies[-1]}")
    torch.save(model, model_path)

    if "curves_path" in args:
        fig, ax = plt.subplots(2, 1, figsize=(20,10))
        ax[0].plot(losses, label="Final loss: %.4f" % losses[-1])
        ax[0].legend()
        ax[0].set_title("BCELoss")
        ax[1].plot(accuracies, label="Final accuracy: %.4f" % accuracies[-1])
        ax[1].legend()
        ax[1].set_title("Accuracy")
        plt.savefig(curves_path)