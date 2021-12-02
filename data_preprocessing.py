import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from neural_config import SERIES_SIZE


def make_data_set(week_sums: np.ndarray, length: int) -> TensorDataset:
    indexes = [list(range(k - length, k)) for k in range(length, week_sums.shape[0])]
    target_idx = list(range(1, len(week_sums)))
    X = []
    Y = []
    for i, s in enumerate(indexes):
        X.append(week_sums[s])
        Y.append(week_sums[target_idx[i]])
    X = np.array(X).astype(np.float32)
    Y = np.array(Y).astype(np.float32)
    X = torch.as_tensor(X)
    Y = torch.as_tensor(Y)
    return TensorDataset(X, Y)


def get_data_loader(path: str = 'data/resampled.csv', user_id: int = 1125811785):
    df = pd.read_csv(path, )
    df = df[df['customer_id'] == user_id]
    df['time'] = pd.to_datetime(df['time'])
    sums = df[df.columns[2:31]]
    scaler = MinMaxScaler(feature_range=(0, 1))
    sums = scaler.fit_transform(sums)
    tensor_data_set = make_data_set(sums, SERIES_SIZE)
    tensor_data_loader = DataLoader(tensor_data_set, batch_size=1)
    return tensor_data_loader,scaler
