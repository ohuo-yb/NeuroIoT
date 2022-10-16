import argparse
import math

import torch.utils.data as Data

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import numpy as np
from model import GridLSTM, GridLSTM_Net, Mul_GridLSTM_Net
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings('ignore')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description="Test for argparse")

parser.add_argument('--epoch', '-e', type=int, default=300)
parser.add_argument('--batch_size', '-b', type=int, default=200)
parser.add_argument('--time_step', '-t', type=int, default=10)
parser.add_argument('--learn_rate', '-l', type=float, default=0.005)
parser.add_argument('--Loss', '-loss', type=str, default="RMSE")

args = parser.parse_args()


# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)  # 固定随机种子（CPU）
    if torch.cuda.is_available():  # 固定随机种子（GPU)
        torch.cuda.manual_seed(seed)  # 为当前GPU设置
        torch.cuda.manual_seed_all(seed)  # 为所有GPU设置
    np.random.seed(seed)  # 保证后续使用random函数时，产生固定的随机数
    torch.backends.cudnn.benchmark = False  # GPU、网络结构固定，可设置为True
    torch.backends.cudnn.deterministic = True  # 固定网络结构


same_seeds(555)


def dataStander(data):
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    return data, scaler


def dataInverse(data, scaler):
    mean = scaler.mean_[1]
    std = scaler.scale_[1]
    data = data * std + mean
    return data


def dataSplit(dataset, timestep):
    data_x = [dataset[i:i + timestep] for i in range(len(dataset) - timestep)]
    data_x = np.array(data_x)
    data_y = [dataset[i + timestep, 1] for i in range(len(dataset) - timestep)]
    data_y = np.array(data_y)
    data_y = data_y.reshape([len(data_y), 1])
    return data_x, data_y


def dataProcess(time_step, history=False, history_timestep=3):
    index = [i for i in range(2, 18)]
    index.extend([21, 22])
    df1 = pd.read_csv('./dataset/SML/NEW-DATA-1.csv', sep=',', usecols=index)
    # 3.14-4.11
    df1 = df1[0:]
    dataset1 = df1[:].values
    dataset1 = dataset1.astype('float32')

    df2 = pd.read_csv('./dataset/SML/NEW-DATA-2.csv', sep=',', usecols=index)
    # 4.18-5.2
    df2 = df2[0:]
    dataset2 = df2[:].values
    dataset2 = dataset2.astype('float32')
    dataset = np.concatenate((dataset1, dataset2), axis=0)
    dataset, scaler = dataStander(dataset)

    data_X, data_Y = dataSplit(dataset, time_step)

    x_train, x_test, y_train, y_test = train_test_split(data_X, data_Y, test_size=0.2)
    train_batch = Data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))  # 为了便于进行batch训练
    test_batch = Data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))  # 为了便于进行batch训练

    # train_set = dataset[:3200]
    #
    # train_set, scaler = dataStander(train_set)
    # validate_set = dataset[3200:3600]
    # validate_set = scaler.transform(validate_set)
    # test_set = dataset[3600:]
    # test_set = scaler.transform(test_set)
    # train_x, train_y = dataSplit(train_set, time_step)
    # validate_x, validate_y = dataSplit(validate_set, time_step)
    # test_x, test_y = dataSplit(test_set, time_step)
    # # return train_x, train_y, validate_x, validate_y, test_x, test_y

    # return train_x, train_y, validate_x, validate_y, test_x, test_y, scaler
    return train_batch, test_batch, scaler


def mape(y_pred, y_true):
    mask = (1 - (y_true == 0))

    loss = np.abs(y_pred - y_true) / (np.abs(y_true) + 1)
    loss *= mask
    non_zero_len = mask.sum()
    return np.sum(loss) / non_zero_len
    # return np.mean(np.abs((y_pred - y_true) / y_true)) * 100


def train(model, dataloader, test_dataloader, epoch, criterion, optimizer, scaler):
    model.train()
    max_loss = 1000
    for i in range(epoch):
        model.train()
        for (train_x, target) in dataloader:
            train_x = train_x.permute(1, 0, 2)
            train_x = train_x.to(device)
            target = target.to(device)
            prediction_t, prediction_d = model(train_x)
            prediction = prediction_t[-1, :, [1]]
            loss = criterion(prediction, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        for (test_x, test_target) in test_dataloader:
            test_x = test_x.permute(1, 0, 2)
            test_x = test_x.to(device)
            test_target = test_target.to(device)
            prediction_t, prediction_d = model(test_x)
            prediction = prediction_t[-1, :, [1]]
            test_loss = criterion(prediction, test_target)

        if test_loss.data < max_loss:
            torch.save(model.state_dict(), "./model/GridLSTM_SML.pt")
            max_loss = test_loss.data
        if i % 10 == 0:
            print("epoch:", i, " | ", "RMSE: train: ", math.sqrt(loss.data), "test: ", math.sqrt(test_loss.data))


def test(model, test_dataloader, criterion):
    model.load_state_dict(torch.load("./model/GridLSTM_SML.pt"))
    model.eval()
    for (test_x, test_target) in test_dataloader:
        test_x = test_x.permute(1, 0, 2)
        test_x = test_x.to(device)
        test_target = test_target.to(device)
        prediction_t, prediction_d = model(test_x)
        prediction = prediction_t[-1, :, [1]]

        test_loss = criterion(prediction, test_target)
        r2_loss = r2_score(prediction.cpu().detach().numpy(), test_target.cpu().detach().numpy())
        MAE_loss = mean_absolute_error(prediction.cpu().detach().numpy(), test_target.cpu().detach().numpy())
        MAPE_loss = mape(prediction.cpu().detach().numpy(), test_target.cpu().detach().numpy())
    print("RMSE: ", math.sqrt(test_loss.data), "| R2:", r2_loss, "| MAE:", MAE_loss, "| MAPE :", MAPE_loss)


def main(args):
    # startTime = datetime.datetime.now()
    batch_size = args.batch_size
    time_step = args.time_step
    lr = args.learn_rate
    epoch = args.epoch

    train_batch, test_batch, scaler = dataProcess(time_step)
    train_loader = Data.DataLoader(dataset=train_batch, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = Data.DataLoader(dataset=test_batch, batch_size=len(test_batch), shuffle=False, drop_last=False)
    model = GridLSTM_Net(input_size=18, hidden_dim=128, batch_size=batch_size).to(device)
    # model = Mul_GridLSTM_Net(input_size=18, hidden_dim=128, batch_size=batch_size).to(device)

    criterion = args.Loss
    if criterion == "RMSE":
        criterion = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(), lr, weight_decay=1e-5)

    train(model, train_loader, test_loader, epoch, criterion, optimizer, scaler)
    test(model, test_loader, criterion)


if __name__ == "__main__":
    main(args)
