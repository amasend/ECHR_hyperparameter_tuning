import hpbandster.core.result as hpres


def store_best_result_and_config(model_path=None, article=None, n_gram=None, tokens=None, flavor=None,
                                 preprocessed=None, pipeline=None):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(model_path)

    # get all executed runs
    all_runs = result.get_all_runs()

    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()

    acc = 0.
    best_run = None
    # Find best scored run
    for run in all_runs:
        if run.info['test mean accuracy'] > acc:
            acc = run.info['test mean accuracy']
            best_run = run
    # Append best_results.csv with data
    with open('best_results.csv', 'a+') as f:
        f.write('{article},{n_gram},{tokens},{model},{flavor},{preprocessed},{pipeline},{test_mean_accuracy},'
                '{test_std_accuracy},{test_mean_f},{test_std_f},'
                '{train_mean_accuracy},{train_std_accuracy},'
                '{train_mean_f},{train_std_f},{train_mean_c_MCC},{train_std_c_MCC},{test_mean_c_MCC},{test_std_c_MCC},'
                '{train_mean_TP},{train_mean_TN},{train_mean_FP},{train_mean_FN},{train_std_TP},{train_std_TN},'
                '{train_std_FP},{train_std_FN},{test_mean_TP},{test_mean_TN},{test_mean_FP},{test_mean_FN},'
                '{test_std_TP},{test_std_TN},{test_std_FP},{test_std_FN},'
                '{config_space}\n'.format(article=article,
                                          n_gram=n_gram,
                                          tokens=tokens,
                                          model=model_path.split('\\')[-2],
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
                                          config_space='\"{}\"'.format(str(id2conf[best_run.config_id]['config']))))