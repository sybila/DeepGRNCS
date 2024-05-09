# DeepGRNCS: Deep learning-based framework for jointly inferring Gene Regulatory Networks across Cell Subpopulations

## Dependencies
- Python == 3.9
- Pytorch == 2.0.0
- scikit-learn == 1.3.0
- numpy == 1.22.3
- pandas == 2.0.3

## Usage
### Dataset
1. For simulated gene expression datasets, regulatory networks between genes are inferred. Prepare the gene expression dataset in the following format:

  |       | Cell1| Cell2|……|
  |-------|------|------|--|
  | Gene1 | 0.046| 1.720|……|
  | Gene2 | 1.817| 0.019|……|
  | …… | ……| ……|……|

2. For real gene expression datasets, regulatory networks between tfs and genes are inferred. Prepare the scRNA-seq dataset (as above), along with information on TFs and genes in the following format:

  | TF|index|
  |-------|------|
  | TF1|1|
  | TF2|3|
  | …… | ……|

  | Gene|index|
  |-------|------|
  | Gene1|0|
  | Gene2|2|
  | …… | ……|

Where 'index' indicates the column index position of the TF (gene) in the scRNA-seq dataset.

### Command to run DeepGRNSC
1. Inferring GRNs based on Gaussian simulated datasets:
  ```
  python main.py --task "DeepGRNCS_demo" --data_file "data/ExpressionData" --save_name "output" --net_number 3
  ```

2. Inferring GRN based on BoolODE simulated dataset:
  ```
  python main.py --task "DeepGRNCS" --data_file "data/ExpressionData" --save_name "output" --net_number 3
  ```

3. Inferring GRNs based on real datasets:
  ```
  python main.py --task "DeepGRNCS_TF" --data_file "data/ExpressionData" --tf_file "data/TF.csv" --gene_file "data/Target.csv"  --save_name "output" --net_number 3
  ```

4. Inferring GRN based on a single real dataset:
  ```
  python main.py --task "DeepGRNCS_TF_Single" --data_file "data/ExpressionData.csv" --tf_file "data/TF.csv" --gene_file "data/Target.csv"  --save_name "output" 
  ```
