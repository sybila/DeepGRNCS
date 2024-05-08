import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score


def load_gene_list(filename):
    gene_list = []
    s = open(filename)
    for line in s:
        gene = line.strip()
        gene_list.append(gene)
    return gene_list


def load_coexpressed_result(filename):
    coexpressed_result = np.zeros((genes_number, genes_number))
    s = open(filename)
    j = 0
    for line in s:
        slice = line.split()
        coexpressed_result[j, :] = slice
        j = j + 1
    s.close()
    coexpressed_result = coexpressed_result.astype(dtype=np.float64)
    return coexpressed_result


def generate_trueEdges(filename):
    refNetwork = pd.read_csv(filename).values
    trueEdgesDF = {}
    for i in range(refNetwork.shape[0]):
        keys = refNetwork[i][0] + ',' + refNetwork[i][1]
        trueEdgesDF[keys] = 1
    return trueEdgesDF


def calculate_predEdgeDict():
    matrix_calculate = np.zeros((genes_number, genes_number))
    for i in range(genes_number):
        mu, std = np.mean(Coexpressed[i]), np.std(Coexpressed[i])
        if std == 0:
            continue
        matrix_calculate[i] = (mu - Coexpressed[i]) / std
    predEdgesDict = {}
    for i in range(genes_number):
        for j in range(genes_number):
            if i == j:
                continue
            keys = gene_list[j] + ',' + gene_list[i]
            predEdgesDict[keys] = matrix_calculate[j][i]
    return predEdgesDict


def calculate_trueEdgeDict():
    trueEdgesDict = {}
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


if __name__ == '__main__':
    gene_list_file = "data/gene_list.txt"
    gene_list = load_gene_list(gene_list_file)
    genes_number = len(gene_list)

    dataset_file_list = ["output/Result" + str(i) + ".txt" for i in range(1, 4)]
    network_file_list = ["data/refNetwork" + str(i) + ".csv" for i in range(1, 4)]
    AUPRC_list = []
    AUROC_list = []

    for i in range(len(dataset_file_list)):
        Coexpressed_file = dataset_file_list[i]
        Coexpressed = load_coexpressed_result(Coexpressed_file)
        trueEdgesDF_file = network_file_list[i]
        labelDict = generate_trueEdges(trueEdgesDF_file)
        predEdgesDict = calculate_predEdgeDict()
        trueEdgesDict = calculate_trueEdgeDict()

        AUPRC, AUROC = computeAUC()
        AUPRC_list.append(AUPRC)
        AUROC_list.append(AUROC)

    print("AUROC: ", end="")
    for auroc in AUROC_list:
        print(format(auroc, '.3f'), end=" ")
    print("")
    print("AUPRC: ", end="")
    for auprc in AUPRC_list:
        print(format(auprc, '.3f'), end=" ")
    print("")
