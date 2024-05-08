# DeepGRNCS: Deep learning-based framework for jointly inferring Gene Regulatory Networks across Cell Subpopulations

## Dependencies
- Python == 3.9
- Pytorch == 2.0.0
- scikit-learn == 1.3.0
- numpy == 1.22.3
- pandas == 2.0.3

## Usage
### For simulated gene expression datasets, regulatory networks between genes are inferred.
1. Prepare the gene expression dataset in the following format:

  |       | Cell1| Cell2|……|
  |-------|------|------|--|
  | Gene1 | 0.046| 1.720|……|
  | Gene2 | 1.817| 0.019|……|
  | …… | ……| ……|……|
3. Command to run DeepGRNSC:

  ```
  python main1.py
  ```
### For real gene expression datasets, inferring TFs on Genes regulatory networks.
1. Prepare the gene expression dataset (as above), along with information on TFs and genes in the following format:

  | Gene|index|
  |-------|------|
  | Gene1|0|
  | Gene2|2|
  | …… | ……|

  | TF|index|
  |-------|------|
  | TF1|1|
  | TF2|3|
  | …… | ……|
3. Command to run DeepGRNSC:
  ```
  python main2.py
  ```
