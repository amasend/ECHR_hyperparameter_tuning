import hpbandster.core.result as hpres
import wget
import zipfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request, json, pprint
import seaborn as sns
import argparse


class Results:
    """Class for results processing."""

    def __init__(self, path, model, grams, tokens, flavor, metric, option):
        sns.set(style="ticks", color_codes=True)
        self.results = {'DNN_IBM': "https://onedrive.live.com/download?cid="
                                   "888296E6B085BF40&resid=888296E6B085BF40%212451&authkey=ALF_fHrkrtCkk2I",
                        'ET': "https://onedrive.live.com/download?cid="
                              "888296E6B085BF40&resid=888296E6B085BF40%212444&authkey=AEBXN0JE1Ru5HmI",
                        'ET_IBM': "https://onedrive.live.com/download?cid="
                                  "888296E6B085BF40&resid=888296E6B085BF40%212447&authkey=AAPG5on77C6Ou-M",
                        'RF': "https://onedrive.live.com/download?cid="
                              "888296E6B085BF40&resid=888296E6B085BF40%212442&authkey=ACZHMcwzA6_C3jA",
                        'RF_IBM': "https://onedrive.live.com/download?cid="
                                  "888296E6B085BF40&resid=888296E6B085BF40%212448&authkey=AN-IxIQRKiUGNLs",
                        'SVM_IBM': "https://onedrive.live.com/download?cid="
                                   "888296E6B085BF40&resid=888296E6B085BF40%212452&authkey=ABylpAEmXkIeo4Y",
                        'XGB': "https://onedrive.live.com/download?cid="
                               "888296E6B085BF40&resid=888296E6B085BF40%212445&authkey=AGLrkyb-sW7KO0E",
                        'XGB_IBM': "https://onedrive.live.com/download?cid="
                                   "888296E6B085BF40&resid=888296E6B085BF40%212449&authkey=AFzuxe2GAz4UyQw"}
        self.worker = path
        self.model = model
        self.grams = grams
        self.tokens = tokens
        self.flavor = flavor
        self.metric = metric
        self.option = option
        self.best_results = None
        self.old_res_data_frame = None
        # self.download_results()
        # self.store_best_result_and_config(metric='test_mean_c_MCC')
        self.store_best_result_and_config(metric=metric)
        # self.show_best_results('test_mean_c_MCC')
        # self.computing_time_vs_runs()
        # self.metric_vs_runs('accuracy')
        # self.show_comparison_old_new_results(metric='test_mean_c_MCC', best=True)
        # self.confusion_matrix(metric='test_mean_c_MCC', normalized=True)

    def download_results(self):
        """Use this function only if you want to download results from OneDrive."""
        wget.download(self.results[self.worker], out='{}.zip'.format(self.worker))
        # Unzip
        zip_ref = zipfile.ZipFile('{}.zip'.format(self.worker), 'r')
        zip_ref.extractall(self.worker)
        zip_ref.close()
        # Delete zipped file
        self.remove_file('{}.zip'.format(self.worker))
        print('{}.zip downloaded and unpacked.'.format(self.worker))

    def store_best_result_and_config(self, preprocessed=None, pipeline=None, metric='test_mean_c_MCC'):
        pd.DataFrame(columns=['article', 'n_gram', 'tokens', 'model', 'flavor', 'preprocessed', 'pipeline',
                              'test_mean_accuracy', 'test_std_accuracy', 'test_mean_f-1', 'test_std_f-1',
                              'train_mean_accuracy', 'train_std_accuracy', 'train_mean_f-1', 'train_std_f-1',
                              'train_mean_c_MCC', 'train_std_c_MCC', 'test_mean_c_MCC', 'test_std_c_MCC',
                              'train_mean_TP', 'train_mean_TN', 'train_mean_FP', 'train_mean_FN', 'train_std_TP',
                              'train_std_TN', 'train_std_FP', 'train_std_FN', 'test_mean_TP', 'test_mean_TN',
                              'test_mean_FP', 'test_mean_FN', 'test_std_TP', 'test_std_TN', 'test_std_FP',
                              'test_std_FN', 'config_space']).to_csv('best_results.csv', index=False)
        if metric == 'test_mean_c_MCC':
            metric = 'test mean c_MCC'
        else:
            metric = 'test mean accuracy'
        for (dirpath, dirnames, filenames) in os.walk(os.path.join('.', self.worker)):
            if filenames:
                try:
                    n_gram = dirpath.split('\\')[2].split('_')[0]
                    tokens = dirpath.split('\\')[3].split('_')[0]
                    article = dirpath.split('\\')[4].split('_')[0]
                    flavor = dirpath.split('\\')[5]
                    model = dirpath.split('\\')[6]
                except IndexError as e:
                    n_gram = dirpath.split('/')[2].split('_')[0]
                    tokens = dirpath.split('/')[3].split('_')[0]
                    article = dirpath.split('/')[4].split('_')[0]
                    flavor = dirpath.split('/')[5]
                    model = dirpath.split('/')[6]
                # load the example run from the log files
                result = hpres.logged_results_to_HBS_result(dirpath)

                # get all executed runs
                all_runs = result.get_all_runs()

                # get the 'dict' that translates config ids to the actual configurations
                id2conf = result.get_id2config_mapping()

                mcc = -1
                best_run = None
                # Find best scored run
                for run in all_runs:
                    if run.info[metric] > mcc and int(run.budget) == 5000:
                        mcc = run.info[metric]
                        best_run = run
                # Append best_results.csv with data
                with open('best_results.csv', 'a+') as f:
                    f.write('{article},{n_gram},{tokens},{model},{flavor},{preprocessed},'
                            '{pipeline},{test_mean_accuracy},'
                            '{test_std_accuracy},{test_mean_f},{test_std_f},'
                            '{train_mean_accuracy},{train_std_accuracy},'
                            '{train_mean_f},{train_std_f},{train_mean_c_MCC},{train_std_c_MCC},'
                            '{test_mean_c_MCC},{test_std_c_MCC},'
                            '{train_mean_TP},{train_mean_TN},{train_mean_FP},{train_mean_FN},'
                            '{train_std_TP},{train_std_TN},'
                            '{train_std_FP},{train_std_FN},{test_mean_TP},{test_mean_TN},'
                            '{test_mean_FP},{test_mean_FN},'
                            '{test_std_TP},{test_std_TN},{test_std_FP},{test_std_FN},'
                            '{config_space}\n'.format(article=article,
                                                      n_gram=n_gram,
                                                      tokens=tokens,
                                                      model=model,
                                                      flavor=flavor,
                                                      preprocessed=preprocessed,
                                                      pipeline=pipeline,
                                                      test_mean_accuracy=best_run.info['test mean accuracy'],
                                                      test_std_accuracy=best_run.info['test std accuracy'],
                                                      test_mean_f=best_run.info['test mean f-1'],
                                                      test_std_f=best_run.info['test std f-1'],
                                                      train_mean_accuracy=best_run.info['train mean accuracy'],
                                                      train_std_accuracy=best_run.info['train std accuracy'],
                                                      train_mean_f=best_run.info['train mean f-1'],
                                                      train_std_f=best_run.info['train std f-1'],
                                                      train_mean_c_MCC=best_run.info['train mean c_MCC'],
                                                      train_std_c_MCC=best_run.info['train std c_MCC'],
                                                      test_mean_c_MCC=best_run.info['test mean c_MCC'],
                                                      test_std_c_MCC=best_run.info['test std c_MCC'],
                                                      train_mean_TP=best_run.info['train mean TP'],
                                                      train_mean_TN=best_run.info['train mean TN'],
                                                      train_mean_FP=best_run.info['train mean FP'],
                                                      train_mean_FN=best_run.info['train mean FN'],
                                                      test_mean_TP=best_run.info['test mean TP'],
                                                      test_mean_TN=best_run.info['test mean TN'],
                                                      test_mean_FP=best_run.info['test mean FP'],
                                                      test_mean_FN=best_run.info['test mean FN'],
                                                      train_std_TP=best_run.info['train std TP'],
                                                      train_std_TN=best_run.info['train std TN'],
                                                      train_std_FP=best_run.info['train std FP'],
                                                      train_std_FN=best_run.info['train std FN'],
                                                      test_std_TP=best_run.info['test std TP'],
                                                      test_std_TN=best_run.info['test std TN'],
                                                      test_std_FP=best_run.info['test std FP'],
                                                      test_std_FN=best_run.info['test std FN'],
                                                      config_space='\"{}\"'.format(
                                                          str(id2conf[best_run.config_id]['config']))))

    def show_best_results(self, metric):
        """Plot metric heatmap per article for particular n_gram and model."""
        self.best_results = pd.read_csv('best_results.csv')
        # print(self.best_results)
        # print(self.best_results.loc[(self.best_results['model'] == model) & (self.best_results['n_gram'] == 1) & (
        #             self.best_results['article'] == '1'), ['tokens', 'flavor', metric]])
        if metric == 'test_mean_accuracy':
            vmin, vmax = 0.5, 1.0
        elif metric == 'test_mean_c_MCC':
            vmin, vmax = 0.0, 1.0
        for n_gram in [1, 2, 3, 4, 5, 6]:
            fig, axis = plt.subplots(6, 2, sharey=True, sharex=True, figsize=(50, 50))
            cbar_ax = fig.add_axes([.91, .3, .03, .4])
            for i, art in enumerate([['1', None], ['2', '3'], ['5', '6'], ['8', '10'], ['11', '13'], ['34', 'p1']]):
                transformed_table = pd.pivot_table(self.best_results.loc[
                                                       (self.best_results['model'] == self.model) & (
                                                               self.best_results['n_gram'] == n_gram) & (
                                                               self.best_results['article'] == art[0]), ['tokens',
                                                                                                         'flavor',
                                                                                                         metric]],
                                                   index='tokens',
                                                   columns='flavor', values=metric)
                print(transformed_table)
                sns.heatmap(transformed_table.T, annot=True, vmin=vmin, vmax=vmax, ax=axis[i, 0], cbar_ax=cbar_ax)
                axis[i, 0].set_title('Article={}'.format(art[0]))
                axis[i, 0].set_ylabel('')
                axis[i, 0].set_xlabel('')
                axis[i, 0].tick_params(axis=u'both', which=u'both', length=0)
                for tick in axis[i, 0].get_yticklabels():
                    tick.set_rotation(0)
                if art[1] is None:
                    axis[i, 1].axis('off')
                    continue
                transformed_table_2 = pd.pivot_table(self.best_results.loc[
                                                         (self.best_results['model'] == self.model) & (
                                                                 self.best_results['n_gram'] == n_gram) & (
                                                                 self.best_results['article'] == art[1]), ['tokens',
                                                                                                           'flavor',
                                                                                                           metric]],
                                                     index='tokens',
                                                     columns='flavor', values=metric)
                print(transformed_table_2)
                sns.heatmap(transformed_table.T, annot=True, vmin=vmin, vmax=vmax, ax=axis[i, 1], cbar_ax=cbar_ax)
                axis[i, 1].set_title('Article={}'.format(art[1]))
                axis[i, 1].set_ylabel('')
                axis[i, 1].set_xlabel('')
                axis[i, 1].tick_params(axis=u'both', which=u'both', length=0)
                for tick in axis[i, 1].get_yticklabels():
                    tick.set_rotation(0)

            plt.subplots_adjust(hspace=0.5)
            fig.suptitle('N_gram: {}\nModel: {}, Metric: {}, '.format(n_gram, self.model, metric))
            fig.text(0.5, 0.04, 'Tokens', ha='center')
            fig.text(0.04, 0.5, 'Flavor', va='center', rotation='vertical')
            plt.show()

    def computing_time_vs_runs(self):
        # TODO: check if there will be some better way to display the results
        """Creates optimization times vs optimization runs graphs."""
        empty_df = pd.DataFrame(columns=['tokens', 'article', 'flavor', 'run_0', 'run_1',
                                         'run_2', 'run_3', 'run_4', 'run_5', 'run_6', 'run_7'])
        # sum_cases = np.sum([951, 1124, 2573, 2292, 6891, 1289, 560, 213, 1090, 136, 1301])
        # # Normalized weights
        # num_cases = {'1': 951/sum_cases, '2': 1124/sum_cases, '3': 2573/sum_cases, '5': 2292/sum_cases,
        #              '6': 6891/sum_cases, '8': 1289/sum_cases, '10': 560/sum_cases, '11': 213/sum_cases,
        #              '13': 1090/sum_cases, '34': 136/sum_cases, 'p1': 1301/sum_cases}

        self.time_vs_run = {1: empty_df,
                            2: empty_df,
                            3: empty_df,
                            4: empty_df,
                            5: empty_df,
                            6: empty_df}
        for (dirpath, dirnames, filenames) in os.walk(os.path.join('.', self.worker)):
            if filenames:
                try:
                    n_gram = dirpath.split('\\')[2].split('_')[0]
                    tokens = dirpath.split('\\')[3].split('_')[0]
                    article = dirpath.split('\\')[4].split('_')[0]
                    flavor = dirpath.split('\\')[5]
                    model = dirpath.split('\\')[6]
                except IndexError as e:
                    n_gram = dirpath.split('/')[2].split('_')[0]
                    tokens = dirpath.split('/')[3].split('_')[0]
                    article = dirpath.split('/')[4].split('_')[0]
                    flavor = dirpath.split('/')[5]
                    model = dirpath.split('/')[6]

                if n_gram == '1' and model == 'random_forest':
                    # load the example run from the log files
                    result = hpres.logged_results_to_HBS_result(dirpath)

                    # get all executed runs
                    all_runs = result.get_all_runs()

                    # get the 'dict' that translates config ids to the actual configurations
                    id2conf = result.get_id2config_mapping()

                    times = {'run_0': [np.inf, 0], 'run_1': [np.inf, 0], 'run_2': [np.inf, 0],
                             'run_3': [np.inf, 0], 'run_4': [np.inf, 0],
                             'run_5': [np.inf, 0], 'run_6': [np.inf, 0], 'run_7': [np.inf, 0],
                             'run_8': [np.inf, 0], 'run_9': [np.inf, 0]}
                    for run in all_runs:
                        if run.time_stamps['started'] < times['run_{}'.format(run.config_id[0])][0]:
                            times['run_{}'.format(run.config_id[0])][0] = run.time_stamps['started']
                        if run.time_stamps['finished'] > times['run_{}'.format(run.config_id[0])][1]:
                            times['run_{}'.format(run.config_id[0])][1] = run.time_stamps['finished']
                    dict_to_append = {'tokens': int(tokens),
                                      'article': article,
                                      'flavor': flavor,
                                      'run_0': times['run_0'][1] - times['run_0'][0],
                                      'run_1': times['run_1'][1] - times['run_1'][0],
                                      'run_2': times['run_1'][1] - times['run_1'][0],
                                      'run_3': times['run_3'][1] - times['run_3'][0],
                                      'run_4': times['run_4'][1] - times['run_4'][0],
                                      'run_5': times['run_5'][1] - times['run_5'][0],
                                      'run_6': times['run_6'][1] - times['run_6'][0],
                                      'run_7': times['run_7'][1] - times['run_7'][0],
                                      'run_8': times['run_8'][1] - times['run_8'][0],
                                      'run_9': times['run_9'][1] - times['run_9'][0],
                                      }
                    self.time_vs_run[int(n_gram)] = \
                        self.time_vs_run[int(n_gram)].append(dict_to_append, ignore_index=True)

        # [self.time_vs_run[i].set_index('tokens', inplace=True) for i in range(1, 7)]
        # [self.time_vs_run[i].set_index('tokens', inplace=True) for i in range(1, 2)]

        for n_gram, value in self.time_vs_run.items():
            # sns.lineplot(x='tokens', y='run_8', hue="flavor", data=value.groupby('article').mean)
            fig, axs = plt.subplots(5, 2)
            for i in range(5):
                sns.barplot(x='tokens', y='run_{}'.format(i), hue="flavor", data=value, ax=axs[i, 0])
                sns.barplot(x='tokens', y='run_{}'.format(i + 5), hue="flavor", data=value, ax=axs[i, 1])
            plt.show()
            # desc_mean = value.loc[value['flavor']=='desc'].groupby('tokens').mean()
            # bow_mean = value.loc[value['flavor']=='bow'].groupby('tokens').mean()
            # desc_bow_mean = value.loc[value['flavor']=='desc+bow'].groupby('tokens').mean()
            #
            # desc_std = value.loc[value['flavor'] == 'desc'].groupby('tokens').std()
            # bow_std = value.loc[value['flavor'] == 'bow'].groupby('tokens').std()
            # desc_bow_std = value.loc[value['flavor'] == 'desc+bow'].groupby('tokens').std()
            #
            # # plt.plot(x=desc_mean.index.values, y=desc_mean['run_0'].values)
            # desc_mean.plot()
            # plt.show()
            # print(value.loc[value['flavor']=='desc'].groupby('tokens').mean())
            # exit()
            fig, axs = plt.subplots(5, 2)
            for i in range(5):
                sns.lineplot(x='tokens', y='run_{}'.format(i), hue="flavor", data=value, ax=axs[i, 0])
                sns.lineplot(x='tokens', y='run_{}'.format(i + 5), hue="flavor", data=value, ax=axs[i, 1])
            plt.show()
            value = value[
                ['tokens', 'article', 'flavor', 'run_9', 'run_8', 'run_7', 'run_6', 'run_5', 'run_4', 'run_3', 'run_2',
                 'run_1', 'run_0']]
            sns.heatmap(value.loc[(value['flavor'] == 'desc') & (value['article'] == '1')].drop(['flavor', 'article'],
                                                                                                axis=1).sort_values(
                by=['tokens']).set_index('tokens').T, annot=True)
            plt.title('Run computation time[s] vs number of tokens. (flavor=desc, article=1)')
            plt.yticks(rotation='horizontal')
            plt.show()

    def metric_vs_runs(self, metric):
        """Creates optimization times vs optimization runs graphs."""
        empty_df = pd.DataFrame(columns=['tokens', 'article', 'flavor', 'run_0', 'run_1',
                                         'run_2', 'run_3', 'run_4', 'run_5', 'run_6', 'run_7'])
        # sum_cases = np.sum([951, 1124, 2573, 2292, 6891, 1289, 560, 213, 1090, 136, 1301])
        # # Normalized weights
        # num_cases = {'1': 951/sum_cases, '2': 1124/sum_cases, '3': 2573/sum_cases, '5': 2292/sum_cases,
        #              '6': 6891/sum_cases, '8': 1289/sum_cases, '10': 560/sum_cases, '11': 213/sum_cases,
        #              '13': 1090/sum_cases, '34': 136/sum_cases, 'p1': 1301/sum_cases}

        self.time_vs_run = {1: empty_df,
                            2: empty_df,
                            3: empty_df,
                            4: empty_df,
                            5: empty_df,
                            6: empty_df}
        for (dirpath, dirnames, filenames) in os.walk(os.path.join('.', self.worker)):
            if filenames:
                try:
                    n_gram = dirpath.split('\\')[2].split('_')[0]
                    tokens = dirpath.split('\\')[3].split('_')[0]
                    article = dirpath.split('\\')[4].split('_')[0]
                    flavor = dirpath.split('\\')[5]
                    model = dirpath.split('\\')[6]
                except IndexError as e:
                    n_gram = dirpath.split('/')[2].split('_')[0]
                    tokens = dirpath.split('/')[3].split('_')[0]
                    article = dirpath.split('/')[4].split('_')[0]
                    flavor = dirpath.split('/')[5]
                    model = dirpath.split('/')[6]

                if n_gram == '1' and model == 'random_forest':
                    # load the example run from the log files
                    result = hpres.logged_results_to_HBS_result(dirpath)

                    # get all executed runs
                    all_runs = result.get_all_runs()

                    # get the 'dict' that translates config ids to the actual configurations
                    id2conf = result.get_id2config_mapping()

                    times = {'run_0': [0, 0], 'run_1': [0, 0], 'run_2': [0, 0],
                             'run_3': [0, 0], 'run_4': [0, 0],
                             'run_5': [0, 0], 'run_6': [0, 0], 'run_7': [0, 0],
                             'run_8': [0, 0], 'run_9': [0, 0]}
                    for run in all_runs:
                        if int(run.budget) == 5000:
                            if run.info['test mean {}'.format(metric)] > times['run_{}'.format(run.config_id[0])][0]:
                                times['run_{}'.format(run.config_id[0])][0] = run.info['test mean {}'.format(metric)]
                                times['run_{}'.format(run.config_id[0])][1] = run.info['test std {}'.format(metric)]
                    dict_to_append = {'tokens': int(tokens),
                                      'article': article,
                                      'flavor': flavor,
                                      'run_0': times['run_0'][0],
                                      'run_1': times['run_1'][0],
                                      'run_2': times['run_1'][0],
                                      'run_3': times['run_3'][0],
                                      'run_4': times['run_4'][0],
                                      'run_5': times['run_5'][0],
                                      'run_6': times['run_6'][0],
                                      'run_7': times['run_7'][0],
                                      'run_8': times['run_8'][0],
                                      'run_9': times['run_9'][0],
                                      }
                    self.time_vs_run[int(n_gram)] = \
                        self.time_vs_run[int(n_gram)].append(dict_to_append, ignore_index=True)

        for n_gram, value in self.time_vs_run.items():
            # fig, axs = plt.subplots(5, 2)
            # for i in range(5):
            #     sns.barplot(x='tokens', y='run_{}'.format(i), hue="flavor", data=value, ax=axs[i, 0])
            #     sns.barplot(x='tokens', y='run_{}'.format(i+5), hue="flavor", data=value, ax=axs[i, 1])
            # plt.show()
            # print(value.set_index(['tokens', 'article', 'flavor']))
            # g = sns.FacetGrid(value.set_index(['article']), row='tokens', col='flavor')
            # g.map(sns.distplot, "run_9")
            # # value.set_index(['tokens', 'article', 'flavor']).plot()
            # plt.show()
            # exit()

            # fig, axs = plt.subplots(5, 2)
            # for i in range(5):
            #     sns.lineplot(x='tokens', y='run_{}'.format(i), hue="flavor", data=value, ax=axs[i, 0])
            #     sns.lineplot(x='tokens', y='run_{}'.format(i + 5), hue="flavor", data=value, ax=axs[i, 1])
            # plt.show()
            value = value[['tokens', 'article', 'flavor', 'run_9', 'run_8', 'run_7',
                           'run_6', 'run_5', 'run_4', 'run_3', 'run_2', 'run_1', 'run_0']]
            if metric == 'accuracy':
                vmin, vmax = 0.6, 1.0
            elif metric == 'c_MCC':
                vmin, vmax = 0.0, 1.0
            for art in ['1', '2', '3', '5', '6', '8', '10', '11', '13', '34', 'p1']:
                fig, axs = plt.subplots(1, 3, figsize=(20, 20), sharey=True)
                for i, flav in enumerate(['desc', 'bow', 'desc+bow']):
                    sns.heatmap(value.loc[(value['flavor'] == '{}'.format(flav)) & (value['article'] == art)].drop(
                        ['flavor', 'article'], axis=1).sort_values(by=['tokens']).set_index('tokens').T,
                                annot=True, ax=axs[i], vmin=vmin, vmax=vmax)
                    axs[i].set_title(
                        'Model {}\n{} per run vs number of tokens.\n(flavor={}, article={}, n_grams={})'.format(model,
                                                                                                                metric,
                                                                                                                flav,
                                                                                                                art,
                                                                                                                n_gram))
                plt.show()

    def show_comparison_old_new_results(self, best=True):
        """Show comparison between old results and new one.
        metric[str] - name of the metric to compare
        best[bool] - indicator to compare with best old results"""
        self.best_results = pd.read_csv('best_results.csv')
        self.load_standard_results()
        self.old_res_data_frame = pd.DataFrame(columns=['article', 'n_gram', 'tokens', 'model', 'flavor',
                                                        'test_mean_accuracy', 'test_std_accuracy', 'test_mean_f-1',
                                                        'train_mean_accuracy', 'train_std_accuracy', 'train_mean_f-1',
                                                        'train_mean_c_MCC', 'test_mean_c_MCC'])
        if self.model == 'random_forest' and not best:
            old_model = 'Random Forest'
            self.transform_old_results([self.model, old_model])
        elif self.model == 'extratrees' and not best:
            old_model = 'Extra Tree'
            self.transform_old_results([self.model, old_model])
        elif self.model == 'xgboost' and not best:
            old_model = 'Gradient Boosting'
            self.transform_old_results([self.model, old_model])
        elif self.model == 'dnn' and not best:
            old_model = 'Neural Net'
            self.transform_old_results([self.model, old_model])
        elif self.model == 'svm' and not best:
            old_model = 'Linear SVC'
            self.transform_old_results([self.model, old_model])
        elif best:
            self.transform_old_results([self.model, None])
        # print(self.old_res_data_frame)

        if self.metric == 'test_mean_accuracy':
            vmin, vmax = 0.5, 1.0
        elif self.metric == 'test_mean_c_MCC':
            vmin, vmax = 0.0, 1.0

        if best:  # Plot the best old result for comparison
            fig = plt.figure(constrained_layout=False, figsize=(50, 100))
            # Defines widths and heights of all subplots on a figure
            widths = [5, 1, 0.8, 5, 1, 0.1]
            heights = [3, 3, 3, 3, 3, 3]
            spec = fig.add_gridspec(ncols=6, nrows=6, width_ratios=widths, height_ratios=heights)
            #  Do not plot figures where is None value (these are dump figures for spacing)
            articles = [['1', '1', None, None, None, None], ['2', '2', None, '3', '3', None],
                        ['5', '5', None, '6', '6', None], ['8', '8', None, '10', '11', None],
                        ['11', '11', None, '13', '13', None], ['34', '34', None, 'p1', 'p1', None]]
            # Define heatbar parameters (horizontal and manual position on a figure)
            cbar_ax = fig.add_axes([.55, .85, .35, .03])
            cbar_kws = {"orientation": "horizontal"}
            for row, row_art in enumerate(articles):
                for col, art in enumerate(row_art):
                    ax = fig.add_subplot(spec[row, col])
                    if art is None:  # Do not plot empty figures (spacing)
                        ax.axis('off')
                        continue
                    if col == 1 or col == 4:  # These are old results
                        # Get results from one article
                        x = self.old_res_data_frame.loc[(self.old_res_data_frame['article'] == art), ['model',
                                                                                                      'tokens',
                                                                                                      'flavor',
                                                                                                      self.metric]]
                        # print(x)
                        x_best = pd.DataFrame(columns=['model', 'tokens', 'flavor', self.metric])
                        # Leave only the best results per flavor (keep model names)
                        for f in ['desc', 'bow', 'desc+bow']:
                            x_part = x.where(x['flavor'] == f)
                            x_best = x_best.append(
                                x_part.loc[x_part[self.metric] == x_part[self.metric].max()].drop_duplicates(
                                    subset=['flavor']))
                        # Merge model names with flavor (we need to display this on a graph)
                        x_best['flavor'] = x_best['flavor'] + ' ' + x_best['model']
                        x_best.drop(['model'], axis=1, inplace=True)
                        # print(x_best)
                        transformed_old = pd.pivot_table(x_best,
                                                         index='tokens',
                                                         columns='flavor',
                                                         values=self.metric).rename(index={5000: '5000_old'})
                        sns.heatmap(transformed_old.T, annot=True, vmin=vmin, vmax=vmax, ax=ax,
                                    cbar=False, cbar_kws=cbar_kws)
                        ax.yaxis.tick_right()
                        ax.set_ylabel('')
                        if row != 5:  # Get rid of all x axis ticks text leave only for the last row
                            ax.set_xlabel('')
                            ax.set_xticklabels([])
                            ax.tick_params(axis=u'x', which=u'both', length=0)
                        for tick in ax.get_yticklabels():
                            tick.set_rotation(0)
                    elif col == 0 or col == 3:  # Plot new results
                        transformed_table = pd.pivot_table(self.best_results.loc[
                                                               (self.best_results['model'] == self.model) & (
                                                                       self.best_results['n_gram'] == self.grams) & (
                                                                       self.best_results['article'] == art), [
                                                                   'tokens',
                                                                   'flavor',
                                                                   self.metric]],
                                                           index='tokens',
                                                           columns='flavor', values=self.metric)
                        # print(transformed_table)
                        sns.heatmap(transformed_table.T, annot=True, vmin=vmin, vmax=vmax, ax=ax,
                                    cbar_ax=cbar_ax, cbar_kws=cbar_kws)
                        ax.set_title('Article={}'.format(art))
                        ax.set_ylabel('')
                        ax.yaxis.tick_right()
                        if row != 5:
                            ax.set_xlabel('')
                            ax.set_xticklabels([])
                            ax.tick_params(axis=u'x', which=u'both', length=0)
                            for tick in ax.get_xticklabels():
                                tick.set_rotation(180)
                        for tick in ax.get_yticklabels():
                            tick.set_rotation(0)
            plt.subplots_adjust(hspace=0.5, wspace=0.5)
            fig.suptitle(
                'Current N_gram: {} --- Model: {} --- Metric: {}\n'
                'Old N_gram: {} --- Model: best --- Metric: {}'.format(self.grams, self.model, self.metric, 4,
                                                                       self.metric))
            fig.text(0.5, 0.04, 'Tokens', ha='center')
            fig.text(0.095, 0.5, 'Flavor', va='center', rotation='vertical')
            plt.show()
        else:
            fig, axis = plt.subplots(6, 2, sharey=True, sharex=True, figsize=(50, 50))
            cbar_ax = fig.add_axes([.91, .3, .03, .4])
            for i, art in enumerate([['1', None], ['2', '3'], ['5', '6'], ['8', '10'], ['11', '13'], ['34', 'p1']]):
                transformed_table = pd.pivot_table(self.best_results.loc[
                                                       (self.best_results['model'] == self.model) & (
                                                               self.best_results['n_gram'] == self.grams) & (
                                                               self.best_results['article'] == art[0]), ['tokens',
                                                                                                         'flavor',
                                                                                                         self.metric]],
                                                   index='tokens',
                                                   columns='flavor', values=self.metric)
                # print(transformed_table)
                transformed_old = pd.pivot_table(self.old_res_data_frame.loc[
                                                     (self.old_res_data_frame['model'] == self.model) & (
                                                             self.old_res_data_frame['article'] == art[0]), ['tokens',
                                                                                                             'flavor',
                                                                                                             self.metric]],
                                                 index='tokens',
                                                 columns='flavor', values=self.metric).rename(index={5000: '5000_old'})
                # print(transformed_old)
                transformed_table = transformed_table.append(transformed_old)
                sns.heatmap(transformed_table.T, annot=True, vmin=vmin, vmax=vmax, ax=axis[i, 0], cbar_ax=cbar_ax)
                axis[i, 0].set_title('Model: {}, Metric: {}, Article={}'.format(self.model,
                                                                                self.metric,
                                                                                art[0]))
                axis[i, 0].set_ylabel('')
                axis[i, 0].set_xlabel('')
                axis[i, 0].tick_params(axis=u'both', which=u'both', length=0)
                for tick in axis[i, 0].get_yticklabels():
                    tick.set_rotation(0)
                if art[1] is None:
                    axis[i, 1].axis('off')
                    continue
                transformed_table_2 = pd.pivot_table(self.best_results.loc[
                                                         (self.best_results['model'] == self.model) & (
                                                                 self.best_results['n_gram'] == self.grams) & (
                                                                 self.best_results['article'] == art[1]), ['tokens',
                                                                                                           'flavor',
                                                                                                           self.metric]],
                                                     index='tokens',
                                                     columns='flavor', values=self.metric)
                # print(transformed_table_2)
                transformed_old = pd.pivot_table(self.old_res_data_frame.loc[
                                                     (self.old_res_data_frame['model'] == self.model) & (
                                                             self.old_res_data_frame['article'] == art[1]), ['tokens',
                                                                                                             'flavor',
                                                                                                             self.metric]],
                                                 index='tokens',
                                                 columns='flavor', values=self.metric).rename(index={5000: '5000_old'})
                # print(transformed_old)
                transformed_table_2 = transformed_table_2.append(transformed_old)
                print(transformed_table_2)
                sns.heatmap(transformed_table.T, annot=True, vmin=vmin, vmax=vmax, ax=axis[i, 1], cbar_ax=cbar_ax)
                axis[i, 1].set_title('Model: {}, Metric: {}, Article={}'.format(self.model,
                                                                                self.metric,
                                                                                art[1]))
                axis[i, 1].set_ylabel('')
                axis[i, 1].set_xlabel('')
                axis[i, 1].tick_params(axis=u'both', which=u'both', length=0)
                for tick in axis[i, 1].get_yticklabels():
                    tick.set_rotation(0)

            plt.subplots_adjust(hspace=0.5)
            fig.suptitle('Current N_gram: {}\nOld N_gram: {}'.format(self.grams, 4))
            fig.text(0.5, 0.04, 'Tokens', ha='center')
            fig.text(0.04, 0.5, 'Flavor', va='center', rotation='vertical')
            plt.show()

    def load_standard_results(self):
        """Loads results from basic experiments."""
        with urllib.request.urlopen(
                "https://raw.githubusercontent.com/aquemy/ECHR-OD_predictions/master/data/output/result_binary.json") as url:
            self.standard_results = json.loads(url.read().decode())

    def transform_old_results(self, models):
        """Takes old result in json format and transforms it into pandas dataframe for further processing."""
        for key, val in self.standard_results.items():
            key_items = key.split(' ')
            article = key_items[1]
            if len(key_items) == 5:  # bow
                flavor = 'bow'
            elif len(key_items) == 6:  # desc
                flavor = 'desc'
            else:  # desc+bow
                flavor = 'desc+bow'
            if models[1] is not None:
                append_dict = {'article': article, 'n_gram': 4, 'tokens': 5000, 'model': models[0], 'flavor': flavor,
                               'test_mean_accuracy': val['methods'][models[1]]['test']['test_acc'],
                               'test_std_accuracy': val['methods'][models[1]]['test']['test_acc_std'],
                               'test_mean_f-1': val['methods'][models[1]]['test']['test_f1_weighted'],
                               'train_mean_accuracy': val['methods'][models[1]]['train']['train_acc'],
                               'train_std_accuracy': val['methods'][models[1]]['train']['train_acc_std'],
                               'train_mean_f-1': val['methods'][models[1]]['train']['train_f1_weighted'],
                               'train_mean_c_MCC': val['methods'][models[1]]['train']['train_mcc'],
                               'test_mean_c_MCC': val['methods'][models[1]]['test']['test_mcc']}
                self.old_res_data_frame = self.old_res_data_frame.append(append_dict, ignore_index=True)
            else:
                for model in val['methods'].keys():
                    append_dict = {'article': article, 'n_gram': 4, 'tokens': 5000, 'model': model,
                                   'flavor': flavor,
                                   'test_mean_accuracy': val['methods'][model]['test']['test_acc'],
                                   'test_std_accuracy': val['methods'][model]['test']['test_acc_std'],
                                   'test_mean_f-1': val['methods'][model]['test']['test_f1_weighted'],
                                   'train_mean_accuracy': val['methods'][model]['train']['train_acc'],
                                   'train_std_accuracy': val['methods'][model]['train']['train_acc_std'],
                                   'train_mean_f-1': val['methods'][model]['train']['train_f1_weighted'],
                                   'train_mean_c_MCC': val['methods'][model]['train']['train_mcc'],
                                   'test_mean_c_MCC': val['methods'][model]['test']['test_mcc']}
                    self.old_res_data_frame = self.old_res_data_frame.append(append_dict, ignore_index=True)

    def confusion_matrix(self, normalized=False):
        """Plot confusion matrices (results can be sorted by metric passed to self.store_best_result_and_config(metric)
        normalized[bool] - normalisation indicator (range [0,1] for all matrices).
        Plot per model, n_gram, tokens, flavor."""
        self.best_results = pd.read_csv('best_results.csv')
        # print(self.best_results.columns)
        fig = plt.figure(constrained_layout=False, figsize=(50, 100))
        # Defines widths and heights of all subplots on a figure
        widths = [4, 4, 0.8, 4, 4]
        heights = [3, 3, 3, 3, 3, 3]
        spec = fig.add_gridspec(ncols=5, nrows=6, width_ratios=widths, height_ratios=heights)
        #  Do not plot figures where is None value (these are dump figures for spacing)
        articles = [['1', '1', None, 'example', None], ['2', '2', None, '3', '3'],
                    ['5', '5', None, '6', '6'], ['8', '8', None, '10', '11'],
                    ['11', '11', None, '13', '13'], ['34', '34', None, 'p1', 'p1']]
        for row, row_art in enumerate(articles):
            for col, art in enumerate(row_art):
                ax = fig.add_subplot(spec[row, col])
                if art is None:  # Do not plot empty figures (spacing)
                    ax.axis('off')
                    continue
                if art == 'example':
                    annot = np.array([['True\nPositive', 'False\nPositive'], ['False\nNegative', 'True\nNegative']])
                    if normalized:
                        example = pd.DataFrame(
                            data={'Violation': [0.3, 0],
                                  'No Violation': [0.7, 1]}).rename(
                            index={0: 'Violation', 1: 'No Violation'})
                        sns.heatmap(example, annot=annot, fmt='', ax=ax, vmin=0, vmax=1)
                    else:
                        example = pd.DataFrame(
                            data={'Violation': [2, 0],
                                  'No Violation': [1, 3]}).rename(
                            index={0: 'Violation', 1: 'No Violation'})
                        sns.heatmap(example, annot=annot, fmt='', ax=ax, vmin=0, vmax=3)
                    ax.xaxis.tick_top()
                    ax.set_title('Example')
                    for tick in ax.get_yticklabels():
                        tick.set_rotation(0)
                    continue
                if col == 0 or col == 3:
                    train = self.best_results.loc[
                        (self.best_results['model'] == self.model) & (self.best_results['n_gram'] == self.grams) & (
                                self.best_results['article'] == art) & (self.best_results['tokens'] == self.tokens) & (
                                self.best_results['flavor'] == self.flavor), ['train_mean_TP',
                                                                              'train_mean_TN', 'train_mean_FP',
                                                                              'train_mean_FN']]
                    # print(train)
                    if normalized:
                        norm_1 = train['train_mean_TN'].values[0] + train['train_mean_FP'].values[0]
                        norm_2 = train['train_mean_FN'].values[0] + train['train_mean_TP'].values[0]
                        train_conf_matrix = pd.DataFrame(
                            data={'Violation': [train['train_mean_TP'].values[0] / norm_2,
                                                train['train_mean_FN'].values[0] / norm_2],
                                  'No Violation': [train['train_mean_FP'].values[0] / norm_1,
                                                   train['train_mean_TN'].values[0] / norm_1],
                                  }).rename(
                            index={0: 'Violation', 1: ' No Violation'})
                        # print(train_conf_matrix)
                        sns.heatmap(train_conf_matrix, annot=True, fmt='.3f', ax=ax, vmin=0, vmax=1,
                                    cbar_kws={"ticks": [0, 0.5, 1]})
                    else:
                        train_conf_matrix = pd.DataFrame(
                            data={'Violation': [train['train_mean_TP'].values[0],
                                                train['train_mean_FN'].values[0]],
                                  'No Violation': [train['train_mean_FP'].values[0],
                                                   train['train_mean_TN'].values[0]],
                                  }).rename(
                            index={0: 'Violation', 1: ' No Violation'})
                        # print(train_conf_matrix)
                        vmax = train.sum(axis=1).values[0]
                        sns.heatmap(train_conf_matrix, annot=True, fmt='.1f', ax=ax, vmin=0, vmax=vmax,
                                    cbar_kws={"ticks": [0, vmax // 2, vmax]})
                    ax.invert_xaxis()
                    ax.set_title('Train Article={}'.format(art))
                    if row != 5:  # Get rid of all x axis ticks text leave only for the last row
                        ax.set_xticklabels([])
                        ax.tick_params(axis=u'x', which=u'both', length=0)
                    if col != 0:
                        ax.set_yticklabels([])
                        ax.tick_params(axis=u'y', which=u'both', length=0)
                    for tick in ax.get_yticklabels():
                        tick.set_rotation(0)
                elif col == 1 or col == 4:
                    test = self.best_results.loc[
                        (self.best_results['model'] == self.model) & (self.best_results['n_gram'] == self.grams) & (
                                self.best_results['article'] == art) & (self.best_results['tokens'] == self.tokens) & (
                                self.best_results['flavor'] == self.flavor), ['test_mean_TP', 'test_mean_TN',
                                                                              'test_mean_FP',
                                                                              'test_mean_FN']]
                    # print(test)
                    if normalized:
                        norm_1 = test['test_mean_TN'].values[0] + test['test_mean_FP'].values[0]
                        norm_2 = test['test_mean_FN'].values[0] + test['test_mean_TP'].values[0]
                        test_conf_matrix = pd.DataFrame(
                            data={'Violation': [test['test_mean_TP'].values[0] / norm_2,
                                                test['test_mean_FN'].values[0] / norm_2],
                                  'No Violation': [test['test_mean_FP'].values[0] / norm_1,
                                                   test['test_mean_TN'].values[0] / norm_1],
                                  }).rename(
                            index={0: 'Violation', 1: ' No Violation'})
                        sns.heatmap(test_conf_matrix, annot=True, fmt='.3f', ax=ax, vmin=0, vmax=1,
                                    cbar_kws={"ticks": [0, 0.5, 1]})
                    else:
                        test_conf_matrix = pd.DataFrame(
                            data={'Violation': [test['test_mean_TP'].values[0],
                                                test['test_mean_FN'].values[0]],
                                  'No Violation': [test['test_mean_FP'].values[0],
                                                   test['test_mean_TN'].values[0]],
                                  }).rename(
                            index={0: 'Violation', 1: ' No Violation'})
                        vmax = test.sum(axis=1).values[0]
                        sns.heatmap(test_conf_matrix, annot=True, fmt='.1f', ax=ax, vmin=0, vmax=vmax,
                                    cbar_kws={"ticks": [0, vmax // 2, vmax]})
                    # print(test_conf_matrix)
                    ax.invert_xaxis()
                    ax.set_title('Test Article={}'.format(art))
                    ax.set_yticklabels([])
                    ax.tick_params(axis=u'y', which=u'both', length=0)
                    if row != 5:  # Get rid of all x axis ticks text leave only for the last row
                        ax.set_xticklabels([])
                        ax.tick_params(axis=u'x', which=u'both', length=0)
        if normalized:
            fig.suptitle(
                'Normalized confusion Matrices by best metric: {}\nN_gram: {}     Model: {}     Tokens: {}'.format(
                    self.metric, self.grams, self.model, self.tokens))
        else:
            fig.suptitle(
                'Confusion Matrices by best metric: {}\nN_gram: {}     Model: {}     Tokens: {}'.format(self.metric,
                                                                                                        self.grams,
                                                                                                        self.model,
                                                                                                        self.tokens))
        fig.text(0.5, 0.04, 'True', ha='center')
        fig.text(0.04, 0.5, 'Predicted', va='center', rotation='vertical')
        plt.subplots_adjust(hspace=0.4)
        plt.show()

    @staticmethod
    def remove_file(file_):
        try:
            os.remove(file_)
            print("File Removed! {}".format(file_))
        except Exception as e:
            print(e)
            exit()

    def chose_option(self):
        if self.option == 'new_best_results':
            self.show_best_results(args.metric)
        elif self.option == 'computing_times':
            self.computing_time_vs_runs()
        elif self.option == 'metrics_vs_runs':
            self.metric_vs_runs('accuracy')
        elif self.option == 'old_vs_new':
            self.show_comparison_old_new_results(best=True)
        elif self.option == 'confusion_matrix':
            self.confusion_matrix(normalized=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main script for result comparison.')
    parser.add_argument('--results_path',
                        type=str,
                        help='Path to results folder.',
                        default='results')
    parser.add_argument('--model',
                        type=str,
                        help='Model name to compare/analysis.',
                        default='random_forest',
                        choices=['extratrees', 'random_forest', 'dnn', 'xgboost', 'svm'])
    parser.add_argument('--n_grams',
                        type=int,
                        help='N-gram to compare.',
                        default=5,
                        choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument('--tokens',
                        type=int,
                        help='Token number to compare.',
                        default=5000,
                        choices=[1000, 5000, 7000, 10000, 30000, 60000, 80000, 100000])
    parser.add_argument('--flavor',
                        type=str,
                        help='Flavor to compare.',
                        default='desc',
                        choices=['desc', 'bow', 'desc+bow'])
    parser.add_argument('--metric',
                        type=str,
                        help='Measurement metric.',
                        default='test_mean_c_MCC',
                        choices=['test_mean_c_MCC', 'test_mean_accuracy'])
    parser.add_argument('--option',
                        type=str,
                        help='What to display.',
                        default='old_vs_new',
                        choices=['new_best_results', 'computing_times', 'metrics_vs_runs',
                                 'old_vs_new', 'confusion_matrix'])
    parser.add_argument('--onedrive',
                        type=bool,
                        help='Use OneDrive storage?',
                        default=False)

    args = parser.parse_args()

    if args.model == 'extratrees':
        results = Results(path=args.results_path, model='extratrees', grams=args.n_grams, tokens=args.tokens,
                          flavor=args.flavor, metric=args.metric, option=args.option)
        results.chose_option()

    elif args.model == 'random_forest':
        results = Results(path=args.results_path, model='random_forest', grams=args.n_grams, tokens=args.tokens,
                          flavor=args.flavor, metric=args.metric, option=args.option)
        results.chose_option()

    elif args.model == 'dnn':
        results = Results(path=args.results_path, model='dnn', grams=args.n_grams, tokens=args.tokens,
                          flavor=args.flavor, metric=args.metric, option=args.option)
        results.chose_option()

    elif args.model == 'xgboost':
        results = Results(path=args.results_path, model='xgboost', grams=args.n_grams, tokens=args.tokens,
                          flavor=args.flavor, metric=args.metric, option=args.option)
        results.chose_option()

    else:
        results = Results(path=args.results_path, model='svm', grams=args.n_grams, tokens=args.tokens,
                          flavor=args.flavor, metric=args.metric, option=args.option)
        results.chose_option()
