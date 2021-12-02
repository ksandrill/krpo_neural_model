import pandas as pd
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader

import drawio
from LstmModel import LstmModel
from data_preprocessing import make_data_set, get_data_loader
from fit_model import train
from neural_config import INPUT_SIZE, NUM_LAYERS, OUTPUT_SIZE, HIDDEN_SIZE, SERIES_SIZE, EPOCH_COUNT, LEARNING_RATE, \
    LINEAR_LAYER_SIZE

MODEL_FILE_NAME = 'model/model_distribution_all_users_sums.pth'


##2:31 - sums, 31:: - counts
def main() -> None:
    model = LstmModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE,
                      linear_layer_size=LINEAR_LAYER_SIZE)
    model.train()
    train_loss = []
    tensor_data_loader, _ = get_data_loader()
    train_loss += train(model, tensor_data_loader, epoch_count=EPOCH_COUNT, lr=LEARNING_RATE)
    torch.save(model.state_dict(), MODEL_FILE_NAME)
    drawio.draw_graph(train_loss, 'epoch', 'mse', 'train_loss')


if __name__ == '__main__':
    main()
