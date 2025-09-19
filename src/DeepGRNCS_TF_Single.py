import os
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.utils_single import MyDataset, Config, Model


class DeepGRNCS_TF_Single:
    def __init__(self, opt):
        self.opt = opt
        self.data_file = opt.data_file
        self.tfs = pd.read_csv(opt.tf_file, index_col=0)['index'].values.astype(np.int32)
        self.genes = pd.read_csv(opt.gene_file, index_col=0)['index'].values.astype(np.int32)

        if not os.path.exists(opt.save_name):
            os.makedirs(opt.save_name)
        self.output_file = os.path.join(opt.save_name, opt.save_file_name)

    def generate_expression_data(self, filename):
        expression_data = pd.read_csv(filename, header='infer', index_col=0)
        expression_data = np.array(expression_data, dtype=np.float64)
        return expression_data

    def generate_tf_data(self, data):
        data_tf = data[self.tfs, :].copy()
        data_tf = np.around(data_tf * 10)
        return data_tf.transpose()

    def generate_ntf_data(self, data):
        data_ntf = data[self.genes, :].copy()
        for i in range(data_ntf.shape[0]):
            row = data_ntf[i]
            row = (np.log10(row / len(row) + 10 ** -4) + 4) / 4
            data_ntf[i] = np.around(row * 10)
        return data_ntf.transpose()

    def generate_tfInof(self):
        tfdict = {}
        for k in range(len(self.tfs)):
            tfdict[self.tfs[k]] = k
        return tfdict

    def generate_index_list(self, length):
        test_size = int(0.1 * length)
        whole_list = range(length)
        test_list = random.sample(whole_list, test_size)
        train_list = [i for i in whole_list if i not in test_list]
        random.shuffle(test_list)
        random.shuffle(train_list)
        return train_list, test_list

    def train_test_split(self, data_X, data_Y, train_list, test_list):
        x_train = data_X[train_list]
        x_test = data_X[test_list]
        y_train = data_Y[train_list]
        y_test = data_Y[test_list]
        return x_train, x_test, y_train, y_test

    def train_model(self):
        expression_data = self.generate_expression_data(self.data_file)
        tfs_dict = self.generate_tfInof()
        print("The number of genes:", expression_data.shape[0])
        print("The number of samples:", expression_data.shape[1])
        data_tf = self.generate_tf_data(expression_data)
        data_ntf = self.generate_ntf_data(expression_data)
        print("The number of tfs:", data_tf.shape[1])
        print("The number of ntfs:", data_ntf.shape[1])
        input_dim = np.int64(data_tf.shape[1])
        output_dim = np.max(np.max(data_ntf))
        output_dim = np.int64(output_dim + 1)
        config = Config(input_dim, output_dim)
        train_list, test_list = self.generate_index_list(expression_data.shape[1])

        print("------------------------" + self.data_file + "---Training Begin!-----------------------")
        coexpressed_result = np.zeros((data_ntf.shape[1], data_tf.shape[1]))
        for j in range(data_ntf.shape[1]):
            print('-------------------------------------', j, '--------------------------------------------')
            iterations = 100
            batch_size = 64
            data_test = data_ntf[:, [j]].copy()
            data_X = data_tf.copy()
            data_medium = data_tf.copy()
            if j in self.tfs:
                tfindex = tfs_dict[j]
                data_X[:, tfindex] = 0
                data_medium[:, tfindex] = 0
            data_Y = data_test.copy()

            # print("1. Preparing input data")
            x_train, x_test, y_train, y_test = self.train_test_split(data_X, data_Y, train_list, test_list)
            train_dataset = MyDataset(x_train, y_train)
            test_dataset = MyDataset(x_test, y_test)
            train_data = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
            test_data = DataLoader(dataset=test_dataset, batch_size=batch_size)

            # print("2. Building models, loss functions and optimizers")
            model = Model(config)
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
                inputs[:, i] = 0
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

        np.savetxt(self.output_file, coexpressed_result)

