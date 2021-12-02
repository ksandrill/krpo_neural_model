import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train(model, data_loader: DataLoader, epoch_count: int, lr: float) -> list[float]:
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_list = []
    for _ in tqdm(range(epoch_count), desc="it's training time!"):
        avg_loss = 0.0
        for i, [X, Y] in enumerate(data_loader):
            outputs = model(X)
            # print(outputs)
            outputs_loss = loss(outputs, Y)
            optimizer.zero_grad()
            outputs_loss.backward()
            optimizer.step()
            # print(" loss:", outputs_loss.item())
            # print(" model output:", outputs.data, "\n real output: ", Y)
            avg_loss += outputs_loss.item()
        avg_loss /= len(data_loader)
        loss_list.append(avg_loss)
    return loss_list
