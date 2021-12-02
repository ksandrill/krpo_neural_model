import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

import drawio
from LstmModel import LstmModel
from data_preprocessing import get_data_loader
from neural_config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE, LINEAR_LAYER_SIZE

MODEL_FILE_NAME = 'model/model_distribution_all_users_sums.pth'


def predict_model(model, data_loader: DataLoader):
    loss = nn.MSELoss()
    loss_list = []
    real_y = []
    predict_y = []
    for i, [X, Y] in enumerate(data_loader):
        outputs = model(X)
        outputs_loss = loss(outputs, Y)
        loss_list.append(outputs_loss.item())
        real_y.append(Y.numpy())
        predict_y.append(outputs.data.numpy())
    return loss_list, real_y, predict_y


if __name__ == '__main__':
    model = LstmModel(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, output_size=OUTPUT_SIZE,
                      linear_layer_size=LINEAR_LAYER_SIZE)
    model.eval()
    model.load_state_dict(torch.load(MODEL_FILE_NAME))
    tensor_data_loader, scaler = get_data_loader()
    loss_list, real_y, predict_y = predict_model(model, tensor_data_loader)
    real_y = np.array(real_y)
    predict_y = np.array(predict_y)
    rmse = np.linalg.norm(real_y - predict_y) / np.sqrt(len(real_y))
    drawio.draw_graph(loss_list, 'week', 'loss', 'test_loss')
    print('rmse error', rmse * 100, " %")
    predict_y = predict_y.squeeze()
    real_y = real_y.squeeze()
    predict_y = scaler.inverse_transform(predict_y)
    real_y = scaler.inverse_transform(real_y)
    for category in range(len(real_y[0])):
        category_list_predict = []
        category_list_real = []

        for elem_idx in range(len(real_y)):
            category_list_predict.append(predict_y[elem_idx, category])
            category_list_real.append(real_y[elem_idx, category])
        drawio.draw_model_real(category_list_predict, category_list_real, 'diff', 'week', 'value', 'blue', 'red')
