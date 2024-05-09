import argparse
from src.DeepGRNCS_TF_Single import DeepGRNCS_TF_Single

parser = argparse.ArgumentParser()
parser.add_argument('--data_file', type=str, help='Input scRNA-seq gene expression file.')
parser.add_argument('--tf_file', type=str, help='Input TFs information file.')
parser.add_argument('--gene_file', type=str, help='Input genes information file.')
parser.add_argument('--save_name', type=str, default='Input the result file.')


if __name__ == '__main__':
    opt = parser.parse_args()
    opt.data_file = "TFs+500/BL--ExpressionData.csv"
    opt.tf_file = "TFs+500/TF.csv"
    opt.gene_file = "TFs+500/Target.csv"
    opt.save_name = "mHSC-L/TFs+500"
    model = DeepGRNCS_TF_Single(opt)
    model.train_model()
