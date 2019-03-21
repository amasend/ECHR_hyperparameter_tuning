# import matplotlib.pyplot as plt
# import hpbandster.core.result as hpres
# import hpbandster.visualization as hpvis
# import argparse
#
#
# parser = argparse.ArgumentParser(description='Module for visualization of '
#                                              'models performance in regards to hyperparameters.')
# parser.add_argument('--shared_directory', type=str, help='Directory within results are stored for particular worker.',
#                     default='.\\results\\dnn\\')
#
# args=parser.parse_args()
#
# # load the example run from the log files
# result = hpres.logged_results_to_HBS_result(args.shared_directory)
#
# # get all executed runs
# all_runs = result.get_all_runs()
#
# # get the 'dict' that translates config ids to the actual configurations
# id2conf = result.get_id2config_mapping()
#
# lcs = result.get_learning_curves()
#
# hpvis.interactive_HBS_plot(lcs, tool_tip_strings=hpvis.default_tool_tips(result, lcs))
#
#
# def realtime_learning_curves(runs):
#     """
#     example how to extract a different kind of learning curve.
#
#     The x values are now the time the runs finished, not the budget anymore.
#     We no longer plot the validation loss on the y axis, but now the test accuracy.
#
#     This is just to show how to get different information into the interactive plot.
#
#     """
#     sr = sorted(runs, key=lambda r: r.budget)
#     lc = list(filter(lambda t: not t[1] is None, [(r.time_stamps['finished'], r.info['test mean accuracy']) for r in sr]))
#     return([lc,])
#
# lcs = result.get_learning_curves(lc_extractor=realtime_learning_curves)
#
# hpvis.interactive_HBS_plot(lcs, tool_tip_strings=hpvis.default_tool_tips(result, lcs))

import pandas as pd

data = pd.read_csv('best_results.csv', index_col=False)
# print(data[['test_mean_accuracy', 'test_mean_c_MCC', 'test_mean_TP']])
# print(data[['test_std_c_MCC']])
# print(data[['test_mean_TN','test_mean_FP','test_mean_FN']])
print(data[['test_mean_c_MCC', 'test_std_c_MCC']])
print(data)
