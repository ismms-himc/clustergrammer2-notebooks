import gzip
from scipy import io
from scipy.sparse import csc_matrix
import pandas as pd
import numpy as np

def filter_barcodes_by_umi(feature_data, feature_type, min_umi=0, max_umi=1e8,
                                      make_sparse=True, zscore_features=False):

    # feature data format
    ########################
    if 'mat' in feature_data[feature_type]:
        mat_csc = feature_data[feature_type]['mat']

        if zscore_features:
            print('*** warning, z-scoring not supported in feature_data format')

        # drop barcodes with fewer than threshold UMI
        ser_sum = mat_csc.sum(axis=0)
        arr_sum = np.asarray(ser_sum[0,:])
        ser_sum = pd.Series(arr_sum[0])
        ser_keep = ser_sum[ser_sum >= min_umi]
        ser_keep = ser_keep[ser_keep <= max_umi]

        # these indexes will be used to filter all features
        keep_indexes = ser_keep.index.tolist()

        # filter barcodes
        barcodes = feature_data[feature_type]['barcodes']
        ser_barcodes = pd.Series(barcodes)
        barcodes_filt = ser_barcodes[keep_indexes].values

        # return Dictionary of DataFrames
        filtered_data = {}
        for inst_feat in feature_data:

            inst_mat = feature_data[inst_feat]['mat']
            mat_filt = inst_mat[:, keep_indexes]
            feature_names = feature_data[inst_feat]['features']

            inst_data = {}
            inst_data['mat'] = mat_filt
            inst_data['barcodes'] = barcodes_filt
            inst_data['features'] = feature_names

            filtered_data[inst_feat] = inst_data

    else:
        # drop barcodes with fewer than threshold UMI
        inst_df = feature_data[feature_type]

        if zscore_features:
            print('z-scoring features')
            net.load_df(inst_df)
            net.normalize(axis='row', norm_type='zscore')
            inst_df = net.export_df()

        ser_sum = inst_df.sum(axis=0)
        ser_keep = ser_sum[ser_sum >= min_umi]
        ser_keep = ser_keep[ser_keep <= max_umi]
        keep_cols = ser_keep.index.tolist()

        # filter data
        filtered_data = {}
        for inst_feat in feature_data:

            filtered_data[inst_feat] = feature_data[inst_feat][keep_cols]

    return filtered_data

def plot_umi_levels(feature_data, feature_type='gex', logy=True, logx=False,
                    figsize=(10,5), min_umi=0, max_umi=1e8, zscore_features=False):
    '''
    This function takes a feature data format or dictionary of DataFrames and plots
    UMI levels
    '''

    if 'mat' in feature_data[feature_type]:
        mat_csc = feature_data[feature_type]['mat']

        if zscore_features:
            print('z-scoring feature_data')
            inst_df = pd.DataFrame(data=mat_csc.todense(), columns=feature_data[feature_type]['barcodes'])

            net.load_df(inst_df)
            net.normalize(axis='row', norm_type='zscore')
            inst_df = net.export_df()

            # sort
            ser_sum = inst_df.sum(axis=0).sort_values(ascending=False)

        else:
            # drop cells with fewer than threshold events
            ser_sum = mat_csc.sum(axis=0)
            arr_sum = np.asarray(ser_sum[0,:])

            # sort
            ser_sum = pd.Series(arr_sum[0], index=feature_data[feature_type]['barcodes']).sort_values(ascending=False)

        # filter
        ser_sum = ser_sum[ser_sum >= min_umi]
        ser_sum = ser_sum[ser_sum <= max_umi]

    else:
        inst_df = feature_data[feature_type]

        if zscore_features:
            print('zscore features')
            net.load_df(inst_df)
            net.normalize(axis='row', norm_type='zscore')
            inst_df = net.export_df()

        # sort
        ser_sum = inst_df.sum(axis=0).sort_values(ascending=False)

        # filter
        ser_sum = ser_sum[ser_sum >= min_umi]
        ser_sum = ser_sum[ser_sum <= max_umi]

    ser_sum.plot(logy=logy, logx=logx, figsize=figsize)
    return ser_sum

def check_feature_data_size(feature_data):
    for inst_feat in feature_data:
        print(inst_feat)
        print(len(feature_data[inst_feat]['features']), len(feature_data[inst_feat]['barcodes']))
        print(feature_data[inst_feat]['mat'].shape, '\n')

def convert_feature_data_to_df_dict(feature_data, make_sparse=True):
    # return Dictionary of DataFrames
    df = {}
    for inst_feat in feature_data:

        inst_mat = feature_data[inst_feat]['mat']
        feature_names = feature_data[inst_feat]['features']
        barcodes = feature_data[inst_feat]['barcodes']

        if make_sparse:
            inst_data = pd.SparseDataFrame(data=inst_mat, index=feature_names, columns=barcodes, default_fill_value=0)
        else:
            inst_data = pd.DataFrame(data=inst_mat.todense(), index=feature_names, columns=barcodes)

        df[inst_feat] = inst_data

    return df

def load_v3_comp_sparse_feat_matrix(inst_path):
    # Read Barcodes
    ###########################
    barcodes_cats = False

    # barcodes
    filename = inst_path + 'barcodes.tsv.gz'
    f = gzip.open(filename, 'rt')
    lines = f.readlines()
    f.close()

    barcodes = []
    for inst_bc in lines:
        inst_bc = inst_bc.strip().split('\t')

        if barcodes_cats == False:
            # remove dash from barcodes if necessary
            if '-' in inst_bc[0]:
                inst_bc[0] = inst_bc[0].split('-')[0]

        barcodes.append(inst_bc[0])

    # parse tuples if necessary
    if barcodes_cats:
        try:
            barcodes = [make_tuple(x) for x in barcodes]
        except:
            pass

    # Load Matrix
    #################
    mat = io.mmread(inst_path + 'matrix.mtx.gz')
    mat_csr = mat.tocsr()

    # Get Indexes of Feature Types
    ##################################
    filename = inst_path + 'features.tsv.gz'
    f = gzip.open(filename, 'rt')
    lines = f.readlines()
    f.close()

    feature_indexes = {}
    feature_lines = {}
    for index in range(len(lines)):

        inst_line = lines[index].strip().split('\t')
        inst_feat = inst_line[2].replace('Gene Expression', 'gex').replace('Antibody Capture', 'adt')


        if inst_feat not in feature_indexes:
            feature_indexes[inst_feat] = []

        feature_indexes[inst_feat].append(index)

    feature_data = {}

    for inst_feat in feature_indexes:
        feature_data[inst_feat] = {}

        feature_data[inst_feat]['barcodes'] = barcodes

        inst_indexes = feature_indexes[inst_feat]

        # Separate feature lists
        ser_lines = pd.Series(lines)
        ser_lines_found = ser_lines[inst_indexes]
        lines_found = ser_lines_found.values.tolist()

        # save feature lines
        feature_lines[inst_feat] = lines_found

        # save as compressed sparse column matrix (for barcode filtering)
        mat_filt = mat_csr[inst_indexes, :].tocsc()

        feature_data[inst_feat]['mat'] = mat_filt

    # Make unique feature names
    for inst_feat in feature_lines:
        feat_lines = feature_lines[inst_feat]
        feat_lines = [x.strip().split('\t') for x in feat_lines]

        # find non-unique initial feature names (add id later if necessary)
        ini_names = [x[1] for x in feat_lines]

        ini_name_count = pd.Series(ini_names).value_counts()
        duplicate_names = ini_name_count[ini_name_count > 1].index.tolist()

        new_names = [x[1] if x[1] not in duplicate_names else x[1] + '_' + x[0] for x in feat_lines]

        # quick hack to clean up names
        new_names = [x.replace('_TotalSeqB', '') for x in new_names]

        feature_data[inst_feat]['features'] = new_names

    return feature_data

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