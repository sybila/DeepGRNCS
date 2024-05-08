import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader


def generate_expression_data(filename):
    expression_data = pd.read_csv(filename, header='infer', index_col=0)
    expression_data = np.array(expression_data, dtype=np.float64)
    return expression_data


def generate_tf_data(data):
    data_tf = data[tfs, :].copy()
    data_tf = np.around(data_tf * 10)
    return data_tf.transpose()


def generate_ntf_data(data):
    data_ntf = data[genes, :].copy()
    for i in range(data_ntf.shape[0]):
        row = data_ntf[i]
        row = (np.log10(row / len(row) + 10 ** -4) + 4) / 4
        data_ntf[i] = np.around(row * 10)
    return data_ntf.transpose()

def generate_tfInof():
    tfdict = {}
    for k in range(len(tfs)):
        tfdict[tfs[k]] = k
    return tfs, tfdict


def generate_multi_data(data_file):
    pair_file_list = []
    for file in dataset_file_list:
        if file != data_file:
            pair_file_list.append(file)
    multi_data = []
    multi_data.append(data_tf)
    for pair_file in pair_file_list:
        pair_expression_data = generate_expression_data(pair_file)
        pair_tf_data = generate_tf_data(pair_expression_data)
        multi_data.append(pair_tf_data)
    return np.array(multi_data)


def generate_index_list(length):
    test_size = int(0.1 * length)
    whole_list = range(length)
    test_list = random.sample(whole_list, test_size)
    train_list = [i for i in whole_list if i not in test_list]
    random.shuffle(test_list)
    random.shuffle(train_list)
    return train_list, test_list


def train_test_split():
    x_train = data_X[:, train_list, :]
    x_test = data_X[:, test_list, :]
    y_train = data_Y[train_list]
    y_test = data_Y[test_list]
    return x_train, x_test, y_train, y_test


class MyDataset:
    def __init__(self, x, y):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)

    def __getitem__(self, item):
        feature_list = []
        for i in range(len(dataset_file_list)):
            feature = self.x[i][item].clone().detach().requires_grad_(True)
            feature_list.append(feature)
        targets = self.y[item].clone().detach().requires_grad_(True)
        return torch.stack(feature_list), targets

    def __len__(self):
        return len(self.y)


class Config(object):
    def __init__(self, input_dim, output_dim):
        self.model_name = 'DeepGRNMS'
        self.dropout = 0.2
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden = 8
        self.learning_rate = 1e-3
        self.nums_network = len(dataset_file_list)


class DeepGRNCS(nn.Module):
    def __init__(self, config):
        super(DeepGRNCS, self).__init__()

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
        out_pair = torch.cat([self.pair_operator(X[i]) for i in range(1, config.nums_network)], 1)
        output = self.FC(torch.cat((out_main, out_pair), 1))
        return output


if __name__ == '__main__':
    network_number = 3
    dataset_file_list = ["data/Subdata" + str(i) + ".csv" for i in range(1, network_number + 1)]
    output_file_list = ["output/Result" + str(i) + ".txt" for i in range(1, network_number + 1)]
    tf_file = "data/TFs.csv"
    gene_file = "data/Genes.csv"
    tfs = pd.read_csv(tf_file, index_col=0)['index'].values.astype(np.int32)
    tfs_dict = generate_tfInof()
    genes = pd.read_csv(gene_file, index_col=0)['index'].values.astype(np.int32)

    for i in range(len(dataset_file_list)):
        data_file = dataset_file_list[i]
        output_file = output_file_list[i]
        expression_data = generate_expression_data(data_file)
        print("The number of genes:", expression_data.shape[0])
        print("The number of samples:", expression_data.shape[1])
        data_tf = generate_tf_data(expression_data)
        data_ntf = generate_ntf_data(expression_data)
        print("The number of tfs:", data_tf.shape[1])
        print("The number of ntfs:", data_ntf.shape[1])
        multi_data = generate_multi_data(data_file)
        train_list, test_list = generate_index_list(expression_data.shape[1])
        input_dim = np.int64(data_tf.shape[1])
        output_dim = np.max(np.max(data_ntf))
        output_dim = np.int64(output_dim + 1)
        config = Config(input_dim, output_dim)

        print("------------------------" + data_file + "---Training Begin!-----------------------")
        coexpressed_result = np.zeros((data_ntf.shape[1], data_tf.shape[1]))
        for j in range(data_ntf.shape[1]):
            print('-------------------------------------', j, '--------------------------------------------')
            iterations = 100
            batch_size = 64
            data_test = data_ntf[:, [j]].copy()
            data_X = multi_data.copy()
            data_X[:, :, j] = 0
            data_medium = multi_data.copy()
            if j in tfs:
                tfindex = tfs_dict[j]
                data_X[:, tfindex] = 0
                data_medium[:, tfindex] = 0
            data_Y = data_test.copy()

            # print("1. Preparing input data")
            x_train, x_test, y_train, y_test = train_test_split()
            train_dataset = MyDataset(x_train, y_train)
            test_dataset = MyDataset(x_test, y_test)
            train_data = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
            test_data = DataLoader(dataset=test_dataset, batch_size=64)

            # print("2. Building models, loss functions and optimizers")
            model = DeepGRNCS(config)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # print("3. Start training")
            losses = []
            acces = []
            eval_losses = []
            eval_acces = []
            for epoch in range(iterations):
                # 3.1 Traing
                train_loss = 0
                train_acc = 0
                for i, data in enumerate(train_data):
                    optimizer.zero_grad()
                    inputs, targets = data
                    inputs = inputs.permute(1, 0, 2)
                    targets = targets.squeeze(1)
                    inputs = Variable(inputs)
                    targets = Variable(targets)
                    inputs = inputs.clone().detach().requires_grad_(True)
                    targets = targets.clone().detach().requires_grad_(True)
                    inputs = inputs.to(torch.float32)
                    targets = targets.to(torch.long)
                    output = model(inputs)
                    loss = criterion(output, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    train_loss = train_loss + loss.item()
                    _, pred = torch.max(output.data, 1)
                    num_correct = (pred == targets).sum().item()
                    train_acc = train_acc + num_correct / inputs[0].shape[0]
                losses.append(train_loss / len(train_data))
                acces.append(train_acc / len(train_data))

                # 3.2 Evaluate
                eval_loss = 0
                eval_acc = 0
                for inputs, targets in test_data:
                    inputs = inputs.permute(1, 0, 2)
                    targets = targets.squeeze(1)
                    inputs = Variable(inputs)
                    targets = Variable(targets)
                    inputs = inputs.clone().detach().requires_grad_(True)
                    targets = targets.clone().detach().requires_grad_(True)
                    inputs = inputs.to(torch.float32)
                    targets = targets.to(torch.long)
                    output = model(inputs)
                    loss = criterion(output, targets)
                    eval_loss = eval_loss + loss.item()
                    _, pred = torch.max(output.data, 1)
                    num_correct = (pred == targets).sum().item()
                    eval_acc = eval_acc + num_correct / inputs[0].shape[0]
                eval_losses.append(eval_loss / len(test_data))
                eval_acces.append(eval_acc / len(test_data))
                if (epoch + 1) % 10 == 0:
                    print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
                          .format(epoch + 1, train_loss / len(train_data), train_acc / len(train_data),
                                  eval_loss / len(test_data), eval_acc / len(test_data)))

            # print("4. Calculating co-expression results")
            accu = []
            for i in range(data_tf.shape[1]):
                inputs = data_medium.copy()
                inputs[:, :, i] = 0
                targets = data_test
                batch_size = targets.shape[0]
                inputs = torch.from_numpy(inputs)
                targets = torch.from_numpy(targets)
                targets = targets.squeeze(1)
                inputs = Variable(inputs)
                targets = Variable(targets)
                inputs = inputs.clone().detach().requires_grad_(True)
                targets = targets.clone().detach().requires_grad_(True)
                inputs = inputs.to(torch.float32)
                targets = targets.to(torch.long)
                output = model(inputs)
                _, pred = torch.max(output.data, 1)
                num_correct = (pred == targets).sum().item()
                accu.append(num_correct / batch_size)
            coexpressed_result[j, :] = accu
        print("--------------------------------Training END!--------------------------------------")

        np.savetxt(output_file, coexpressed_result)
