# Usage of https://github.com/aquemy/ECHR-OD_loader from different directory
import sys
sys.path.insert(0, 'C:\\Users\\Amadeusz\\Documents\\GitHub\\ECHR-OD_loader\\')
sys.path.insert(0, 'C:\\Users\\Amadeusz\\Documents\\GitHub\\ECHR-OD_loader\\echr\\')
import echr

from xgboost import XGBClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np
import pprint

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)


class XGBoostWorker(Worker):
    """BOHB: ROBUST AND EFFICIENT HYPERPARAMETER OPTIMIZATION AT SCALE
    Keras framework class for DNN hyperparameter tuning."""

    def __init__(self, **kwargs):
        """Initialize DNN with:
        - batch size for each worker,
        - number of classes to predict,
        - scroring method for cross-validation
        - dataset for evaluation"""

        super().__init__(**kwargs)
        self.scoring = ['f1', 'accuracy']

        # Loads dataset
        dataset = echr.binary.get_dataset(article='11', flavors=[echr.Flavor.desc])  # Define the dataset model
        self.X, self.y = dataset.load()  # Load in memory the dataset
        # Make process pipeline
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=config['num_pca']))
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=900))
        std_clf = make_pipeline(StandardScaler())
        self.X = std_clf.fit_transform(self.X.toarray())

    def compute(self, config: object, budget, working_directory, *args, **kwargs):
        """Compute method for each BOHB run evaluation.
        Defines data preprocessing pipeline and keras DNN model for particular run.
        data -> StandardScaler -> (any preprocess technique) -> k-fold_cv with Keras_model -> train_and_test_results
        """

        # Wraps Keras model into sklearn classifier (this is a need for use with cross-validation)
        model = XGBClassifier(max_depth=config['max_depth'], learning_rate=config['lr'],
                              n_estimators=config['n_estimators'],
                              objective='binary:logistic',
                              booster='gbtree',
                              gamma=config['gamma'], min_child_weight=config['min_ch_w'],
                              max_delta_step=config['max_d_s'],
                              subsample=config['subsample'], colsample_bytree=config['cols_bytree'],
                              colsample_bylevel=config['cols_bylevel'],
                              reg_alpha=config['reg_alpha'], reg_lambda=config['reg_lambda'])

        # k-fld test, produce estimated score per algorithm based on training and test datasets
        kfold = StratifiedKFold(n_splits=int(budget), random_state=7)
        cv_results = cross_validate(model, self.X, self.y,
                                    cv=kfold, scoring=self.scoring,
                                    return_train_score=True)

        # Returns information about BOHB loss function and all the computed metrics from DNN
        return ({
            'loss': 1-np.mean(cv_results['test_accuracy']), # remember: HpBandSter always minimizes!
            'info': {'test mean accuracy': np.mean(cv_results['test_accuracy']),
                     'test std accuracy': np.std(cv_results['test_accuracy']),
                     'train mean accuracy': np.mean(cv_results['train_accuracy']),
                     'train std accuracy': np.std(cv_results['train_accuracy']),
                     'train mean f-1': np.mean(cv_results['train_f1']),
                     'train std f-1': np.std(cv_results['train_f1']),
                     'test mean f-1': np.mean(cv_results['test_f1']),
                     'test std f-1': np.std(cv_results['test_f1'])
                     }
            })

    @staticmethod
    def get_configspace():
        """:return: ConfigurationsSpace-Object
        Here is the main place to create particular hyperparameters to tune.
        Particular hyperparameter should be defined as:
        hyperparameter = type_of_parameter(name, lower_range, upper_range, default_value, logging)
        add.hyperparameter([hyperparameter])
        """

        cs = CS.ConfigurationSpace()

        # num_pca = CSH.UniformIntegerHyperparameter('num_pca', lower=850, upper=930, default_value=900, log=True)
        # cs.add_hyperparameters([num_pca])

        n_estimators = CSH.UniformIntegerHyperparameter('n_estimators', lower=1, upper=1000,
                                                        default_value=500, log=True)
        lr = CSH.UniformFloatHyperparameter('lr',  lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        max_depth = CSH.UniformIntegerHyperparameter('max_depth', lower=100, upper=1000,
                                                     default_value=200, log=True)
        gamma = CSH.UniformFloatHyperparameter('gamma', lower=0.1, upper=100, default_value=1, log=True)
        min_ch_w = CSH.UniformFloatHyperparameter('min_ch_w', lower=0.1, upper=100, default_value=1, log=True)
        max_d_s = CSH.UniformFloatHyperparameter('max_d_s', lower=0.01, upper=100, default_value=0.01, log=True)
        subsample = CSH.UniformFloatHyperparameter('subsample', lower=0.001, upper=1., default_value=1, log=True)
        cols_bytree = CSH.UniformFloatHyperparameter('cols_bytree', lower=0.001, upper=1., default_value=1, log=True)
        cols_bylevel = CSH.UniformFloatHyperparameter('cols_bylevel', lower=0.001, upper=1., default_value=1, log=True)
        reg_alpha = CSH.UniformFloatHyperparameter('reg_alpha', lower=0.001, upper=1., default_value=0.001, log=True)
        reg_lambda = CSH.UniformFloatHyperparameter('reg_lambda', lower=0.001, upper=1., default_value=1, log=True)

        cs.add_hyperparameters([n_estimators, lr, max_depth, gamma, min_ch_w, max_d_s, subsample, cols_bylevel,
                                cols_bytree, reg_alpha, reg_lambda])

        return cs



# This part is only used to validation of above code in one run.
# keras_optimize.py implements whole BOHB searching space procedure
if __name__ == "__main__":
    worker = XGBoostWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=1, working_directory='.')
    pp = pprint.PrettyPrinter()
    pp.pprint(res)
