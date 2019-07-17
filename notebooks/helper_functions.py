def load_comp_10x_feature_mtx_to_sparse_df(inst_path):
  '''
  Loads gene expression data from 10x in sparse matrix format and returns a
  Pandas dataframe
  '''

  import pandas as pd
  from scipy import io
  from scipy import sparse
  from ast import literal_eval as make_tuple

  # matrix
  mat = io.mmread( inst_path + 'matrix.mtx.gz')


  # features
  import gzip
  filename = inst_path + 'features.tsv.gz'
  f = gzip.open(filename, 'rt')
  lines = f.readlines()
  f.close()


  # add unique id only to duplicate genes
  ini_genes = []
  for inst_line in lines:
      inst_line = inst_line.strip().split()
      if len(inst_line) > 1:
        inst_gene = inst_line[1]
      else:
        inst_gene = inst_line[0]
      ini_genes.append(inst_gene)

  gene_name_count = pd.Series(ini_genes).value_counts()
  duplicate_genes = gene_name_count[gene_name_count > 1].index.tolist()

  dup_index = {}
  genes = []
  for inst_row in ini_genes:

    # add index to non-unique genes
    if inst_row in duplicate_genes:

      # calc_non-unque index
      if inst_row not in dup_index:
        dup_index[inst_row] = 1
      else:
        dup_index[inst_row] = dup_index[inst_row] + 1

      new_row = inst_row + '_' + str(dup_index[inst_row])

    else:
      new_row = inst_row

    genes.append(new_row)

  # barcodes
  import gzip
  filename = inst_path + 'barcodes.tsv.gz'
  f = gzip.open(filename, 'rt')
  lines = f.readlines()
  f.close()

  barcodes = []
  for inst_bc in lines:

      inst_bc = inst_bc.strip().split('\t')

      # remove dash from barcodes if necessary
      if '-' in inst_bc[0]:
        inst_bc[0] = inst_bc[0].split('-')[0]

      barcodes.append(inst_bc[0])

  # parse tuples if necessary
  try:
      barcodes = [make_tuple(x) for x in barcodes]
  except:
      pass

  try:
      genes = [make_tuple(x) for x in genes]
  except:
      pass

  # # make dataframe
  # df = pd.SparseDataFrame(mat, index=genes, columns=barcodes, default_fill_value=0)

  # return df

  return genes, barcodes, mat