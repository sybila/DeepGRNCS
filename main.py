import argparse
import numpy as np
import random
import torch

from src.DeepGRNCS_TF import DeepGRNCS_TF
from src.DeepGRNCS import DeepGRNCS
from src.DeepGRNCS_TF_Single import DeepGRNCS_TF_Single
from src.DeepGRNCS_demo import DeepGRNCS_demo

parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default='DeepGRNCS',
                    help='Determine which task to run. Select from (DeepGRNCS,DeepGRNCS_TF,DeepGRNCS_TF_Single)')
parser.add_argument('--data_file', type=str, help='Input scRNA-seq gene expression file.')
parser.add_argument('--tf_file', type=str, help='Input TFs information file.')
parser.add_argument('--gene_file', type=str, help='Input genes information file.')
parser.add_argument('--save_name', type=str, default='Input the result file.')
parser.add_argument('--net_number', type=int, default=1, help='Input the number of subpopulations.')
parser.add_argument('--seed', type=int, help = 'Seed for controlling the randomness of the run')


def set_seed(seed: int) -> None:
    # Have to set seed per library used to make sure it's the same
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


if __name__ == '__main__':
    opt = parser.parse_args()

    if opt.seed is not None:
        set_seed(opt.seed)

    if opt.task == 'DeepGRNCS':
        model = DeepGRNCS(opt)
        model.train_model()
    elif opt.task == 'DeepGRNCS_TF':
        model = DeepGRNCS_TF(opt)
        model.train_model()
    elif opt.task == 'DeepGRNCS_TF_Single':
        model = DeepGRNCS_TF_Single(opt)
        model.train_model()
    elif opt.task == 'DeepGRNCS_demo':
        model = DeepGRNCS_demo(opt)
        model.train_model()
