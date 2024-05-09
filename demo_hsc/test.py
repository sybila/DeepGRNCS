import argparse
from src.DeepGRNCS import DeepGRNCS

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, help='Input scRNA-seq gene expression file.')
parser.add_argument('--save_name', type=str, default='Input the result file.')
parser.add_argument('--net_number', type=int, default=1, help='Input the number of subpopulations.')


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.data_file = "data/ExpressionData"
    opt.save_name = "output"
    opt.net_number = 3
    model = DeepGRNCS(opt)
    model.train_model()

