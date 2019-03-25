import os
from distutils.dir_util import copy_tree


def make_dir(dir_):
    try:
        os.makedirs(dir_)
    except Exception as e:
        print(e)


def move_results_to_storage(n_gram=None, tokens=None, article=None, flavor=None, model=None):
    path = './all_results/{n_gram}_gram'.format(n_gram=n_gram)
    make_dir(path)
    path = './all_results/{n_gram}_gram/{tokens}_tokens'.format(n_gram=n_gram, tokens=tokens)
    make_dir(path)
    path = './all_results/{n_gram}_gram/{tokens}_tokens/{article}_article'.format(n_gram=n_gram, tokens=tokens, article=article)
    make_dir(path)
    path = './all_results/{n_gram}_gram/{tokens}_tokens/{article}_article/{flavor}'.format(n_gram=n_gram,
                                                                                   tokens=tokens,
                                                                                   article=article,
                                                                                   flavor=flavor)
    make_dir(path)
    copy_tree('./results/{}/{}'.format(flavor, model), '{}/{}'.format(path, model))