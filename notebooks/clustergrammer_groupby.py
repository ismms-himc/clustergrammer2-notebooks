import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from copy import deepcopy
from sklearn.metrics import pairwise_distances, roc_curve, auc
from scipy.spatial.distance import pdist
from copy import deepcopy
from sklearn.metrics import confusion_matrix
from copy import deepcopy
import numpy as np
import random
from itertools import combinations
import matplotlib.pyplot as plt

def sim_same_and_diff_category_samples(df, cat_index=1, dist_type='cosine',
                                       equal_var=False, plot_roc=True,
                                       precalc_dist=False):
    '''
    Calculate the similarity of samples from the same and different categories. The
    cat_index gives the index of the category, where 1 in the first category.
    '''

    cols = df.columns.tolist()

    if type(precalc_dist) == bool:
        # compute distnace between rows (transpose to get cols as rows)
        dist_arr = 1 - pdist(df.transpose(), metric=dist_type)
    else:
        dist_arr = precalc_dist

    # generate sample names with categories
    sample_combos = list(combinations(range(df.shape[1]),2))
    sample_names = [(cols[x[0]][cat_index] + '_' + cols[x[1]][cat_index], cols[x[0]][cat_index], cols[x[1]][cat_index])
                    for x in sample_combos]

    ser_dist = pd.Series(data=dist_arr, index=sample_names)

    # find same-cat sample comparisons
    same_cat = [x for x in sample_names if x[1] == x[2]]

    # find diff-cat sample comparisons
    diff_cat = [x for x in sample_names if x[1] != x[2]]

    # make series of same and diff category sample comparisons
    ser_same = ser_dist[same_cat]
    ser_same.name = 'Same Category'
    ser_diff = ser_dist[diff_cat]
    ser_diff.name = 'Different Category'

    sim_dict = {}
    sim_dict['same'] = ser_same
    sim_dict['diff'] = ser_diff

    pval_dict = {}
    ttest_stat, pval_dict['ttest'] = ttest_ind(ser_diff, ser_same, equal_var=equal_var)

    ttest_stat, pval_dict['mannwhitney'] = mannwhitneyu(ser_diff, ser_same)

    # calc AUC
    true_index = list(np.ones(sim_dict['same'].shape[0]))
    false_index = list(np.zeros(sim_dict['diff'].shape[0]))
    y_true = true_index + false_index

    true_val = list(sim_dict['same'].get_values())
    false_val = list(sim_dict['diff'].get_values())
    y_score = true_val + false_val

    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    inst_auc = auc(fpr, tpr)

    if plot_roc:
        plt.figure()
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
        plt.figure(figsize=(10,10))

        print('AUC', inst_auc)

    roc_data = {}
    roc_data['true'] = y_true
    roc_data['score'] = y_score
    roc_data['fpr'] = fpr
    roc_data['tpr'] = tpr
    roc_data['thresholds'] = thresholds
    roc_data['auc'] = inst_auc


    return sim_dict, pval_dict, roc_data


def generate_signatures(df_ini, category_level, pval_cutoff=0.05,
                        num_top_dims=False, verbose=True, equal_var=False):

    ''' Generate signatures for column categories '''

    df_t = df_ini.transpose()

    # remove columns with constant values
    df_t = df_t.loc[:, (df_t != df_t.iloc[0]).any()]

    df = row_tuple_to_multiindex(df_t)

    cell_types = sorted(list(set(df.index.get_level_values(category_level).tolist())))

    keep_genes = []
    keep_genes_dict = {}
    gene_pval_dict = {}
    all_fold_info = {}

    for inst_ct in cell_types:

        inst_ct_mat = df.xs(key=inst_ct, level=category_level)
        inst_other_mat = df.drop(inst_ct, level=category_level)

        # save mean values and fold change
        fold_info = {}
        fold_info['cluster_mean'] = inst_ct_mat.mean()
        fold_info['other_mean'] = inst_other_mat.mean()
        fold_info['log2_fold'] = fold_info['cluster_mean']/fold_info['other_mean']
        fold_info['log2_fold'] = fold_info['log2_fold'].apply(np.log2)
        all_fold_info[inst_ct] = fold_info

        inst_stats, inst_pvals = ttest_ind(inst_ct_mat, inst_other_mat, axis=0, equal_var=equal_var)

        ser_pval = pd.Series(data=inst_pvals, index=df.columns.tolist()).sort_values()

        if num_top_dims == False:
            ser_pval_keep = ser_pval[ser_pval < pval_cutoff]
        else:
            ser_pval_keep = ser_pval[:num_top_dims]

        gene_pval_dict[inst_ct] = ser_pval_keep

        inst_keep = ser_pval_keep.index.tolist()
        keep_genes.extend(inst_keep)
        keep_genes_dict[inst_ct] = inst_keep

    keep_genes = sorted(list(set(keep_genes)))

    df_gbm = df.groupby(level=category_level).mean().transpose()
    cols = df_gbm.columns.tolist()
    new_cols = []
    for inst_col in cols:
        new_col = (inst_col, category_level + ': ' + inst_col)
        new_cols.append(new_col)
    df_gbm.columns = new_cols

    df_sig = df_gbm.ix[keep_genes]

    if len(keep_genes) == 0 and verbose:
        print('found no informative dimensions')

    df_gene_pval = pd.concat(gene_pval_dict, axis=1)

    return df_sig, keep_genes_dict, df_gene_pval, all_fold_info

def predict_cats_from_sigs(df_data_ini, df_sig_ini, dist_type='cosine', predict_level='Predict Category',
                           truth_level=1, unknown_thresh=-1):
    ''' Predict category using signature '''

    keep_rows = df_sig_ini.index.tolist()
    data_rows = df_data_ini.index.tolist()

    common_rows = list(set(data_rows).intersection(keep_rows))

    df_data = deepcopy(df_data_ini.ix[common_rows])
    df_sig = deepcopy(df_sig_ini.ix[common_rows])

    # calculate sim_mat of df_data and df_sig
    cell_types = df_sig.columns.tolist()
    barcodes = df_data.columns.tolist()
    sim_mat = 1 - pairwise_distances(df_sig.transpose(), df_data.transpose(), metric=dist_type)
    df_sim = pd.DataFrame(data=sim_mat, index=cell_types, columns=barcodes).transpose()

    # get the top column value (most similar signature)
    df_sim_top = df_sim.idxmax(axis=1)

    # get the maximum similarity of a cell to a cell type definition
    max_sim = df_sim.max(axis=1)

    unknown_cells = max_sim[max_sim < unknown_thresh].index.tolist()

    # assign unknown cells (need category of same name)
    df_sim_top[unknown_cells] = 'Unknown'

    # add predicted category name to top list
    top_list = df_sim_top.get_values()
    top_list = [ predict_level + ': ' + x[0] if type(x) is tuple else predict_level + ': ' + x  for x in top_list]

    # add cell type category to input data
    df_cat = deepcopy(df_data)
    cols = df_cat.columns.tolist()
    new_cols = []

    # check whether the columns have the true category available
    has_truth = False
    if type(cols[0]) is tuple:
        has_truth = True

    if has_truth:
        new_cols = [tuple(list(a) + [b]) for a,b in zip(cols, top_list)]
    else:
        new_cols = [tuple([a] + [b]) for a,b in zip(cols, top_list)]

    # transfer new categories
    df_cat.columns = new_cols

    # keep track of true and predicted labels
    y_info = {}
    y_info['true'] = []
    y_info['pred'] = []

    if has_truth:
        y_info['true'] = [x[truth_level].split(': ')[1] for x in cols]
        y_info['pred'] = [x.split(': ')[1] for x in top_list]

    return df_cat, df_sim.transpose(), y_info

def confusion_matrix_and_correct_series(y_info):
    ''' Generate confusion matrix from y_info '''


    a = deepcopy(y_info['true'])
    true_count = dict((i, a.count(i)) for i in set(a))

    a = deepcopy(y_info['pred'])
    pred_count = dict((i, a.count(i)) for i in set(a))

    sorted_cats = sorted(list(set(y_info['true'] + y_info['pred'])))
    conf_mat = confusion_matrix(y_info['true'], y_info['pred'], sorted_cats)
    df_conf = pd.DataFrame(conf_mat, index=sorted_cats, columns=sorted_cats)

    total_correct = np.trace(df_conf)
    total_pred = df_conf.sum().sum()
    fraction_correct = total_correct/float(total_pred)

    # calculate ser_correct
    correct_list = []
    cat_counts = df_conf.sum(axis=1)
    all_cols = df_conf.columns.tolist()
    for inst_cat in all_cols:
        inst_correct = df_conf[inst_cat].loc[inst_cat] / cat_counts[inst_cat]
        correct_list.append(inst_correct)

    ser_correct = pd.Series(data=correct_list, index=all_cols)

    populations = {}
    populations['true'] = true_count
    populations['pred'] = pred_count

    return df_conf, populations, ser_correct, fraction_correct


def compare_performance_to_shuffled_labels(df_data, category_level, num_shuffles=100,
                                           random_seed=99, pval_cutoff=0.05, dist_type='cosine',
                                           num_top_dims=False, predict_level='Predict Category',
                                           truth_level=1, unknown_thresh=-1, equal_var=False,
                                           performance_type='prediction'):
    random.seed(random_seed)

    perform_list = []
    num_shuffles = num_shuffles

    # pre-calculate the distance matrix (similarity matrix) if necessary
    if performance_type == 'cat_sim_auc':
        dist_arr = 1 - pdist(df_data.transpose(), metric=dist_type)

    for inst_run in range(num_shuffles + 1):

        cols = df_data.columns.tolist()
        rows = df_data.index.tolist()
        mat = df_data.get_values()

        shuffled_cols = deepcopy(cols)
        random.shuffle(shuffled_cols)

        # do not perform shuffling the first time to confirm that we get the same
        # results as the unshuffled dataaset
        if inst_run == 0:
            df_shuffle = deepcopy(df_data)
        else:
            df_shuffle = pd.DataFrame(data=mat, columns=shuffled_cols, index=rows)

        # generate signature on shuffled data
        df_sig, keep_genes, keep_genes_dict, fold_info = generate_signatures(df_shuffle,
                                                           category_level,
                                                           pval_cutoff=pval_cutoff,
                                                           num_top_dims=num_top_dims,
                                                           equal_var=equal_var)

        # predictive performance
        if performance_type == 'prediction':

            # predict categories from signature
            df_pred_cat, df_sig_sim, y_info = predict_cats_from_sigs(df_shuffle, df_sig,
                dist_type=dist_type, predict_level=predict_level, truth_level=truth_level,
                unknown_thresh=unknown_thresh)

            # calc confusion matrix and performance
            df_conf, populations, ser_correct, fraction_correct = confusion_matrix_and_correct_series(y_info)

            # store performances of shuffles
            if inst_run > 0:
                perform_list.append(fraction_correct)
            else:
                real_performance = fraction_correct
                print('performance (fraction correct) of unshuffled: ' + str(fraction_correct))

        elif performance_type == 'cat_sim_auc':

            # predict categories from signature
            sim_dict, pval_dict, roc_data = sim_same_and_diff_category_samples(df_shuffle,
                cat_index=1, plot_roc=False, equal_var=equal_var, precalc_dist=dist_arr)

            # store performances of shuffles
            if inst_run > 0:
                perform_list.append(roc_data['auc'])
            else:
                real_performance = roc_data['auc']
                print('performance (category similarity auc) of unshuffled: ' + str(roc_data['auc']))

    perform_ser = pd.Series(perform_list)

    in_top_fraction = perform_ser[perform_ser > real_performance].shape[0]/num_shuffles
    print('real data performs in the top ' + str(in_top_fraction*100) + '% of shuffled labels\n')

    return perform_ser


def box_scatter_plot(df, group, columns=False, rand_seed=100, alpha=0.5,
    dot_color='red', num_row=None, num_col=1, figsize=(10,10),
    start_title='Variable Measurements Across', end_title='Groups',
    group_list=False):

    from scipy import stats
    import pandas as pd

    import matplotlib.pyplot as plt
    # %matplotlib inline

    if columns == False:
        columns = df.columns.tolist()

    plt.figure(figsize=figsize)
    figure_title = start_title + ' ' + group + ' ' + end_title
    plt.suptitle(figure_title, fontsize=20)

    # list of arranged dataframes
    dfs = {}

    for col_num in range(len(columns)):
        column = columns[col_num]
        plot_id = col_num + 1

        # group by column name or multiIndex name
        if group in df.columns.tolist():
            grouped = df.groupby(group)
        else:
            grouped = df.groupby(level=group)

        names, vals, xs = [], [] ,[]

        if type(column) is tuple:
            column_title = column[0]
        else:
            column_title = column

        for i, (name, subdf) in enumerate(grouped):

            names.append(name)

            inst_ser = subdf[column]

            column_name = column_title + '-' + str(name)

            inst_ser.name = column_name
            vals.append(inst_ser)

            np.random.seed(rand_seed)
            xs.append(np.random.normal(i+1, 0.04, subdf.shape[0]))

        ax = plt.subplot(num_row, num_col, plot_id)

        plt.boxplot(vals, labels=names)

        ngroup = len(vals)

        clevels = np.linspace(0., 1., ngroup)

        for x, val, clevel in zip(xs, vals, clevels):

            plt.subplot(num_row, num_col, plot_id)
            plt.scatter(x, val, c=dot_color, alpha=alpha)


        df_arranged = pd.a(vals, axis=1)

        # anova
        anova_data = [df_arranged[col].dropna() for col in df_arranged]
        f_val, pval = stats.f_oneway(*anova_data)

        if pval < 0.01:
            ax.set_title(column_title + ' P-val: ' + '{:.2e}'.format(pval))
        else:
            pval = round(pval * 100000)/100000
            ax.set_title(column_title + ' P-val: ' + str(pval))

        dfs[column] = df_arranged

    return dfs

def rank_cols_by_anova_pval(df, group, columns=False, rand_seed=100, alpha=0.5, dot_color='red', num_row=None, num_col=1,
                     figsize=(10,10)):

    from scipy import stats
    import numpy as np
    import pandas as pd

    # import matplotlib.pyplot as plt
    # %matplotlib inline

    if columns == False:
        columns = df.columns.tolist()

    # plt.figure(figsize=figsize)

    # list of arranged dataframes
    dfs = {}

    pval_list = []

    for col_num in range(len(columns)):
        column = columns[col_num]
        plot_id = col_num + 1

        # group by column name or multiIndex name
        if group in df.columns.tolist():
            grouped = df.groupby(group)
        else:
            grouped = df.groupby(level=group)

        names, vals, xs = [], [] ,[]

        if type(column) is tuple:
            column_title = column[0]
        else:
            column_title = column

        for i, (name, subdf) in enumerate(grouped):
            names.append(name)

            inst_ser = subdf[column]

            column_name = column_title + '-' + str(name)

            inst_ser.name = column_name
            vals.append(inst_ser)

            np.random.seed(rand_seed)
            xs.append(np.random.normal(i+1, 0.04, subdf.shape[0]))


        ngroup = len(vals)

        df_arranged = pd.concat(vals, axis=1)

        # anova
        anova_data = [df_arranged[col].dropna() for col in df_arranged]
        f_val, pval = stats.f_oneway(*anova_data)

        pval_list.append(pval)

    pval_ser = pd.Series(data=pval_list, index=columns)
    pval_ser = pval_ser.sort_values(ascending=True)

    return pval_ser


def row_tuple_to_multiindex(df):
    import pandas as pd

    from copy import deepcopy
    df_mi = deepcopy(df)
    rows = df_mi.index.tolist()
    titles = []
    for inst_part in rows[0]:

        if ': ' in inst_part:
            inst_title = inst_part.split(': ')[0]
        else:
            inst_title = 'Name'
        titles.append(inst_title)

    new_rows = []
    for inst_row in rows:
        inst_row = list(inst_row)
        new_row = []
        for inst_part in inst_row:
            if ': ' in inst_part:
                inst_part = inst_part.split(': ')[1]
            new_row.append(inst_part)
        new_row = tuple(new_row)
        new_rows.append(new_row)

    df_mi.index = new_rows

    df_mi.index = pd.MultiIndex.from_tuples(df_mi.index, names=titles)

    return df_mi
