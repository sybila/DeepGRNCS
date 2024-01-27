import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def load_gene_list(filename):
    data = pd.read_csv(filename, header="infer", index_col=0)
    genes = data.index.tolist()
    return genes


def load_coexpressed_result(filename):
    coexpressed_result = np.zeros((len(targets), len(tfs)))
    s = open(filename)
    j = 0
    for line in s:
        seperate = line.split()
        coexpressed_result[j, :] = seperate
        j = j + 1
    s.close()
    coexpressed_result = coexpressed_result.astype(dtype=np.float64)
    return coexpressed_result


def calculate_trueEdgeDict():
    trueEdgesDict = {}
    labelDict = {}
    for i in range(label.shape[0]):
        keys = label[i][0] + '|' + label[i][1]
        labelDict[keys] = 1
    for key in predEdgesDict.keys():
        if key in labelDict.keys():
            trueEdgesDict[key] = 1
        else:
            trueEdgesDict[key] = 0
    return trueEdgesDict


def computeAUC():
    outDF = pd.DataFrame([trueEdgesDict, predEdgesDict]).T
    outDF.columns = ['TrueEdges', 'PredEdges']

    AUROC = roc_auc_score(y_true=outDF['TrueEdges'], y_score=outDF['PredEdges'])
    AUPRC = average_precision_score(y_true=outDF['TrueEdges'], y_score=outDF['PredEdges'])
    return AUPRC, AUROC


def calculate_predEdgeDict(score):
    matrix_calculate = np.zeros((len(targets), len(tfs)))
    for i in range(len(targets)):
        mu, std = np.mean(Coexpressed[i]), np.std(Coexpressed[i])
        if std < 0.001:
            continue
        matrix_calculate[i] = (mu - Coexpressed[i]) / std
    predEdgesDict = {}
    for i in range(len(targets)):
        for j in range(len(tfs)):
            if tfs[j] == targets[i] or matrix_calculate[i][j] < score:
                continue
            keys = tfs[j] + '|' + targets[i]
            predEdgesDict[keys] = matrix_calculate[i][j]
    return predEdgesDict


if __name__ == '__main__':
    label_file = "TFs+500/Label.csv"
    tfs_file =  "TFs+500/TF.csv"
    targets_file = "TFs+500/Target.csv"
    label = pd.read_csv(label_file, index_col=0).values.astype(np.str_)
    tfs = pd.read_csv(tfs_file, index_col=0)['index'].values.astype(np.str_)
    targets = pd.read_csv(targets_file, index_col=0)['index'].values.astype(np.str_)

    Coexpressed_file = "TFs+500/Result.txt"
    Coexpressed = load_coexpressed_result(Coexpressed_file)
    predEdgesDict = calculate_predEdgeDict(3)
    trueEdgesDict = calculate_trueEdgeDict()

    AUPRC, AUROC = computeAUC()

    print("AUROC: ", format(AUROC, '.3f'))
    print("AUPRC: ", format(AUPRC, '.3f'))

