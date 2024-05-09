import torch
import torch.nn as nn

class MyDataset:
    def __init__(self, x, y, number):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.number = number

    def __getitem__(self, item):
        feature_list = []
        for i in range(self.number):
            feature = self.x[i][item].clone().detach().requires_grad_(True)
            feature_list.append(feature)
        targets = self.y[item].clone().detach().requires_grad_(True)
        return torch.stack(feature_list), targets

    def __len__(self):
        return len(self.y)


class Config(object):
    def __init__(self, input_dim, output_dim, number):
        self.model_name = 'DeepGRNCS'
        self.dropout = 0.2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = 8
        self.learning_rate = 1e-3
        self.nums_network = number


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config

        self.main_operator = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden),
            nn.BatchNorm1d(config.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(config.hidden, config.hidden),
            nn.BatchNorm1d(config.hidden),
            nn.ReLU(inplace=True)
        )

        self.pair_operator = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden),
            nn.BatchNorm1d(config.hidden),
            nn.ReLU(inplace=True)
        )

        self.FC = nn.Sequential(
            nn.Linear(config.hidden * config.nums_network, config.output_dim),
            nn.BatchNorm1d(config.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.output_dim, config.output_dim)
        )

    def forward(self, X):
        out_main = self.main_operator(X[0])
        out_pair = torch.cat([self.pair_operator(X[i]) for i in range(1, self.config.nums_network)], 1)
        output = self.FC(torch.cat((out_main, out_pair), 1))
        return output
