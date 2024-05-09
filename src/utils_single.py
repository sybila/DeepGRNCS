import torch
import torch.nn as nn

class MyDataset:
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, item):
        feature = self.x[item].clone().detach().requires_grad_(True)
        targets = self.y[item].clone().detach().requires_grad_(True)
        return feature, targets

    def __len__(self):
        return len(self.y)


class Config(object):
    def __init__(self, input_dim, output_dim):
        self.model_name = 'DeepGRNCS'
        self.dropout = 0.2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = 8
        self.learning_rate = 1e-3


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()

        self.hidden = config.hidden
        self.main_operator = nn.Sequential(
            nn.Linear(config.input_dim, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden, self.hidden),
            nn.BatchNorm1d(self.hidden),
            nn.ReLU(inplace=True)
        )

        self.FC = nn.Sequential(
            nn.Linear(self.hidden, config.output_dim),
            nn.BatchNorm1d(config.output_dim),
            nn.ReLU(inplace=True),
            nn.Linear(config.output_dim, config.output_dim)
        )

    def forward(self, X):
        out_main = self.main_operator(X)
        output = self.FC(out_main)
        return output
