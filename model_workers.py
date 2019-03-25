# Usage of https://github.com/aquemy/ECHR-OD_loader from different directory
import sys
sys.path.insert(0, '/dev/ECHR-OD_loader/')
sys.path.insert(0, '/dev/ECHR-OD_loader/echr/')
import echr
try:
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras import backend as K
except:
    raise ImportError("For this example you need to install keras.")
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import logging
logging.basicConfig(level=logging.DEBUG)


def compute_tn(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn


def compute_fp(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fp


def compute_fn(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn


def compute_tp(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tp


def custom_mcc(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        denominator = 1
    numerator = tp * tn - fp * fn
    return numerator / denominator


_SEED_ = 7
ACC = make_scorer(accuracy_score)
F1 = make_scorer(f1_score)
TN = make_scorer(compute_tn)
FP = make_scorer(compute_fp)
FN = make_scorer(compute_fn)
TP = make_scorer(compute_tp)
c_MCC = make_scorer(custom_mcc)


class XGBoostWorker(Worker):
    """BOHB: ROBUST AND EFFICIENT HYPERPARAMETER OPTIMIZATION AT SCALE
    XGBoost worker hyperparameter tuning."""

    def __init__(self, **kwargs):
        """Initialize DNN with:
        - scroring method for cross-validation
        - dataset for evaluation"""
        super_kwargs = dict(kwargs)
        del super_kwargs['article']
        del super_kwargs['flavor']
        super().__init__(**super_kwargs)

        self.scoring = {'f1': F1, 'accuracy': ACC, 'TN': TN, 'TP': TP, 'FN': FN, 'FP': FP, 'c_MCC': c_MCC}

        # Loads dataset
        if kwargs['flavor'] == 'desc':
            dataset = echr.binary.get_dataset(article=kwargs['article'], flavors=[echr.Flavor.desc])
        elif kwargs['flavor'] == 'bow':
            dataset = echr.binary.get_dataset(article=kwargs['article'], flavors=[echr.Flavor.bow])
        else:
            dataset = echr.binary.get_dataset(article=kwargs['article'])
        self.X, self.y = dataset.load()  # Load in memory the dataset
        # Make process pipeline
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=config['num_pca']))
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=900))
        # std_clf = make_pipeline(StandardScaler())
        # self.X = std_clf.fit_transform(self.X.toarray())

    def compute(self, config: object, budget, working_directory, *args, **kwargs):
        """Compute method for each BOHB run evaluation.
        Defines data preprocessing pipeline and keras DNN model for particular run.
        data -> StandardScaler -> (any preprocess technique) -> k-fold_cv -> train_and_test_results
        """

        # Wraps Keras model into sklearn classifier (this is a need for use with cross-validation)
        model = XGBClassifier(max_depth=config['max_depth'], learning_rate=config['lr'],
                              n_estimators=config['n_estimators'], gamma=config['gamma'])
                              # objective='binary:logistic',
                              # booster='gbtree',
                              # gamma=config['gamma'], min_child_weight=config['min_ch_w'],
                              # max_delta_step=config['max_d_s'],
                              # subsample=config['subsample'], colsample_bytree=config['cols_bytree'],
                              # colsample_bylevel=config['cols_bylevel'],
                              # reg_alpha=config['reg_alpha'], reg_lambda=config['reg_lambda'])

        # k-fld test, produce estimated score per algorithm based on training and test datasets
        if int(budget) == 555:
            b = 5
        elif int(budget) == 1666:
            b = 7
        else:
            b = 10
        kfold = StratifiedKFold(n_splits=b, random_state=_SEED_)
        cv_results = cross_validate(model, self.X, self.y,
                                    cv=kfold, scoring=self.scoring,
                                    return_train_score=True)

        # Returns information about BOHB loss function and all the computed metrics from DNN
        return ({
            'loss': 1-np.mean(cv_results['test_c_MCC']),  # remember: HpBandSter always minimizes!
            'info': {'test mean accuracy': np.mean(cv_results['test_accuracy']),
                     'test std accuracy': np.std(cv_results['test_accuracy']),
                     'train mean accuracy': np.mean(cv_results['train_accuracy']),
                     'train std accuracy': np.std(cv_results['train_accuracy']),
                     'train mean f-1': np.mean(cv_results['train_f1']),
                     'train std f-1': np.std(cv_results['train_f1']),
                     'test mean f-1': np.mean(cv_results['test_f1']),
                     'test std f-1': np.std(cv_results['test_f1']),
                     'train mean TP': np.mean(cv_results['train_TP']),
                     'train mean TN': np.mean(cv_results['train_TN']),
                     'train mean FP': np.mean(cv_results['train_FP']),
                     'train mean FN': np.mean(cv_results['train_FN']),
                     'test mean TP': np.mean(cv_results['test_TP']),
                     'test mean TN': np.mean(cv_results['test_TN']),
                     'test mean FP': np.mean(cv_results['test_FP']),
                     'test mean FN': np.mean(cv_results['test_FN']),
                     'train std TP': np.std(cv_results['train_TP']),
                     'train std TN': np.std(cv_results['train_TN']),
                     'train std FP': np.std(cv_results['train_FP']),
                     'train std FN': np.std(cv_results['train_FN']),
                     'test std TP': np.std(cv_results['test_TP']),
                     'test std TN': np.std(cv_results['test_TN']),
                     'test std FP': np.std(cv_results['test_FP']),
                     'test std FN': np.std(cv_results['test_FN']),
                     'test mean c_MCC': np.mean(cv_results['test_c_MCC']),
                     'test std c_MCC': np.std(cv_results['test_c_MCC']),
                     'train mean c_MCC': np.mean(cv_results['train_c_MCC']),
                     'train std c_MCC': np.std(cv_results['train_c_MCC']),
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
                                                        default_value=100, log=True)
        lr = CSH.UniformFloatHyperparameter('lr',  lower=0.001, upper=0.9, default_value=0.1, log=True)
        max_depth = CSH.UniformIntegerHyperparameter('max_depth', lower=1, upper=1000,
                                                     default_value=3, log=True)
        gamma = CSH.UniformFloatHyperparameter('gamma', lower=0.1, upper=100, default_value=1, log=True)
        # min_ch_w = CSH.UniformFloatHyperparameter('min_ch_w', lower=0.1, upper=100, default_value=1, log=True)
        # max_d_s = CSH.UniformFloatHyperparameter('max_d_s', lower=0.01, upper=100, default_value=0.01, log=True)
        # subsample = CSH.UniformFloatHyperparameter('subsample', lower=0.001, upper=1., default_value=1, log=True)
        # cols_bytree = CSH.UniformFloatHyperparameter('cols_bytree', lower=0.001, upper=1., default_value=1, log=True)
        # cols_bylevel = CSH.UniformFloatHyperparameter('cols_bylevel', lower=0.001, upper=1., default_value=1, log=True)
        # reg_alpha = CSH.UniformFloatHyperparameter('reg_alpha', lower=0.001, upper=1., default_value=0.001, log=True)
        # reg_lambda = CSH.UniformFloatHyperparameter('reg_lambda', lower=0.001, upper=1., default_value=1, log=True)

        cs.add_hyperparameters([n_estimators, lr, max_depth, gamma])#, gamma, min_ch_w, max_d_s, subsample, cols_bylevel,
                                # cols_bytree, reg_alpha, reg_lambda])

        return cs


class RandomForestWorker(Worker):
    """BOHB: ROBUST AND EFFICIENT HYPERPARAMETER OPTIMIZATION AT SCALE
    Randowm Forest worker hyperparameter tuning."""

    def __init__(self, **kwargs):
        """Initialize DNN with:
        - scroring method for cross-validation
        - dataset for evaluation"""
        super_kwargs = dict(kwargs)
        del super_kwargs['article']
        del super_kwargs['flavor']
        super().__init__(**super_kwargs)

        self.scoring = {'f1': F1, 'accuracy': ACC, 'TN': TN, 'TP': TP, 'FN': FN, 'FP': FP, 'c_MCC': c_MCC}

        # Loads dataset
        if kwargs['flavor'] == 'desc':
            dataset = echr.binary.get_dataset(article=kwargs['article'], flavors=[echr.Flavor.desc])
        elif kwargs['flavor'] == 'bow':
            dataset = echr.binary.get_dataset(article=kwargs['article'], flavors=[echr.Flavor.bow])
        else:
            dataset = echr.binary.get_dataset(article=kwargs['article'])
        self.X, self.y = dataset.load()  # Load in memory the dataset
        # Make process pipeline
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=config['num_pca']))
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=900))
        # std_clf = make_pipeline(StandardScaler())
        # self.X = std_clf.fit_transform(self.X.toarray())

    def compute(self, config: object, budget, working_directory, *args, **kwargs):
        """Compute method for each BOHB run evaluation.
        Defines data preprocessing pipeline and keras DNN model for particular run.
        data -> StandardScaler -> (any preprocess technique) -> k-fold_cv-> train_and_test_results
        """

        # Wraps Keras model into sklearn classifier (this is a need for use with cross-validation)
        # print(config)
        model = RandomForestClassifier(n_estimators=config['n_estimators'], criterion=config['criterion'],
                                       max_depth=config['max_depth'], random_state=_SEED_,
                                       min_samples_split=config['min_sample_split'],)
                                       # min_samples_leaf=config['min_sample_leaf'],
                                       # max_features=config['max_features'], max_leaf_nodes=config['max_leaf_nodes'],
                                       # min_impurity_decrease=config['min_impur_dist'])

        # k-fld test, produce estimated score per algorithm based on training and test datasets
        if int(budget) == 555:
            b = 5
        elif int(budget) == 1666:
            b = 7
        else:
            b = 10
        kfold = StratifiedKFold(n_splits=b, random_state=_SEED_)
        cv_results = cross_validate(model, self.X, self.y,
                                    cv=kfold, scoring=self.scoring,
                                    return_train_score=True)
        # print(cv_results['test_c_MCC'])

        # Returns information about BOHB loss function and all the computed metrics from DNN
        return ({
            'loss': 1-np.mean(cv_results['test_c_MCC']), # remember: HpBandSter always minimizes!
            'info': {'test mean accuracy': np.mean(cv_results['test_accuracy']),
                     'test std accuracy': np.std(cv_results['test_accuracy']),
                     'train mean accuracy': np.mean(cv_results['train_accuracy']),
                     'train std accuracy': np.std(cv_results['train_accuracy']),
                     'train mean f-1': np.mean(cv_results['train_f1']),
                     'train std f-1': np.std(cv_results['train_f1']),
                     'test mean f-1': np.mean(cv_results['test_f1']),
                     'test std f-1': np.std(cv_results['test_f1']),
                     'train mean TP': np.mean(cv_results['train_TP']),
                     'train mean TN': np.mean(cv_results['train_TN']),
                     'train mean FP': np.mean(cv_results['train_FP']),
                     'train mean FN': np.mean(cv_results['train_FN']),
                     'test mean TP': np.mean(cv_results['test_TP']),
                     'test mean TN': np.mean(cv_results['test_TN']),
                     'test mean FP': np.mean(cv_results['test_FP']),
                     'test mean FN': np.mean(cv_results['test_FN']),
                     'train std TP': np.std(cv_results['train_TP']),
                     'train std TN': np.std(cv_results['train_TN']),
                     'train std FP': np.std(cv_results['train_FP']),
                     'train std FN': np.std(cv_results['train_FN']),
                     'test std TP': np.std(cv_results['test_TP']),
                     'test std TN': np.std(cv_results['test_TN']),
                     'test std FP': np.std(cv_results['test_FP']),
                     'test std FN': np.std(cv_results['test_FN']),
                     'test mean c_MCC': np.mean(cv_results['test_c_MCC']),
                     'test std c_MCC': np.std(cv_results['test_c_MCC']),
                     'train mean c_MCC': np.mean(cv_results['train_c_MCC']),
                     'train std c_MCC': np.std(cv_results['train_c_MCC']),
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

        n_estimators = CSH.UniformIntegerHyperparameter('n_estimators', lower=1, upper=500,
                                                        default_value=100, log=True)
        criterion = CSH.CategoricalHyperparameter('criterion', ['gini', 'entropy'])
        max_depth = CSH.UniformIntegerHyperparameter('max_depth', lower=100, upper=1000,
                                                     default_value=None, log=True)
        min_sample_split = CSH.UniformIntegerHyperparameter('min_sample_split', lower=2, upper=30,
                                                            default_value=2, log=True)
        # min_sample_leaf = CSH.UniformIntegerHyperparameter('min_sample_leaf', lower=1, upper=100,
        #                                                    default_value=50, log=True)
        # max_features = CSH.CategoricalHyperparameter('max_features', ['auto', 'sqrt', 'log2'])
        # max_leaf_nodes = CSH.UniformIntegerHyperparameter('max_leaf_nodes', lower=10, upper=1000,
        #                                                   default_value=500, log=True)
        # min_impur_dist = CSH.UniformFloatHyperparameter('min_impur_dist', lower=0.1, upper=1.0,
        #                                                 default_value=0.5, log=True)

        cs.add_hyperparameters([n_estimators, criterion, max_depth, min_sample_split])
        # cs.add_hyperparameters([n_estimators, criterion, max_depth,
        #                         min_sample_split, min_sample_leaf,
        #                         max_features, max_leaf_nodes, min_impur_dist])

        return cs


class KerasWorker(Worker):
    """BOHB: ROBUST AND EFFICIENT HYPERPARAMETER OPTIMIZATION AT SCALE
    Keras framework class for DNN hyperparameter tuning."""

    def __init__(self, **kwargs):
        """Initialize DNN with:
        - batch size for each worker,
        - number of classes to predict,
        - scroring method for cross-validation
        - dataset for evaluation"""
        super_kwargs = dict(kwargs)
        del super_kwargs['article']
        del super_kwargs['flavor']
        super().__init__(**super_kwargs)

        self.batch_size = 64
        self.num_classes = 2
        self.scoring = {'f1': F1, 'accuracy': ACC, 'TN': TN, 'TP': TP, 'FN': FN, 'FP': FP, 'c_MCC': c_MCC}


        # Loads dataset
        if kwargs['flavor'] == 'desc':
            dataset = echr.binary.get_dataset(article=kwargs['article'], flavors=[echr.Flavor.desc])
        elif kwargs['flavor'] == 'bow':
            dataset = echr.binary.get_dataset(article=kwargs['article'], flavors=[echr.Flavor.bow])
        else:
            dataset = echr.binary.get_dataset(article=kwargs['article'])
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

        def keras_model(config=config, shape=self.X.shape[1]):
            """For each BOHB need to define separate Keras model with different parameters:
            - dense neurons number for each dense layer
            - dropout value
            - optimization algorithm
            - learning rate and/or parameters for aptimization algorith"""

            model = Sequential()
            model.add(Dense(config['num_fc_units_1'], input_dim=shape, activation=config['activation']))
            model.add(Dropout(config['dropout_rate_1']))
            model.add(Dense(config['num_fc_units_2'], activation=config['activation']))
            model.add(Dropout(config['dropout_rate_2']))
            model.add(Dense(config['num_fc_units_3'], activation=config['activation']))
            model.add(Dropout(config['dropout_rate_3']))
            model.add(Dense(config['num_fc_units_4'], activation=config['activation']))
            model.add(Dropout(config['dropout_rate_4']))
            model.add(Dense(2, activation='softmax'))

            # Choose an optimizer to use
            if config['optimizer'] == 'Adam':
                optimizer = keras.optimizers.Adam(lr=config['lr'])
            elif config['optimizer'] == 'Adadelta':
                optimizer = keras.optimizers.Adadelta(lr=config['lr'])
            elif config['optimizer'] == 'RMSprop':
                optimizer = keras.optimizers.RMSprop(lr=config['lr'])
            else:
                optimizer = keras.optimizers.SGD(lr=config['lr'], momentum=config['sgd_momentum'])

            # Compile the model with binary crossentropy as a loss function and binary accuracy as a metric
            model.compile(loss=keras.losses.binary_crossentropy,
                          optimizer=optimizer,
                          metrics=['binary_accuracy'])
            return model

        # Wraps Keras model into sklearn classifier (this is a need for use with cross-validation)
        model = MyKerasClassifier(build_fn=keras_model,
                                  epochs=int(budget),
                                  batch_size=self.batch_size,
                                  verbose=0)

        # k-fld test, produce estimated score per algorithm based on training and test datasets
        kfold = StratifiedKFold(n_splits=10, random_state=_SEED_)
        cv_results = cross_validate(model, self.X, self.y,
                                    cv=kfold, scoring=self.scoring,
                                    return_train_score=True)

        # Returns information about BOHB loss function and all the computed metrics from DNN
        return ({
            'loss': 1-np.mean(cv_results['test_c_MCC']), # remember: HpBandSter always minimizes!
            'info': {'test mean accuracy': np.mean(cv_results['test_accuracy']),
                     'test std accuracy': np.std(cv_results['test_accuracy']),
                     'train mean accuracy': np.mean(cv_results['train_accuracy']),
                     'train std accuracy': np.std(cv_results['train_accuracy']),
                     'train mean f-1': np.mean(cv_results['train_f1']),
                     'train std f-1': np.std(cv_results['train_f1']),
                     'test mean f-1': np.mean(cv_results['test_f1']),
                     'test std f-1': np.std(cv_results['test_f1']),
                     'train mean TP': np.mean(cv_results['train_TP']),
                     'train mean TN': np.mean(cv_results['train_TN']),
                     'train mean FP': np.mean(cv_results['train_FP']),
                     'train mean FN': np.mean(cv_results['train_FN']),
                     'test mean TP': np.mean(cv_results['test_TP']),
                     'test mean TN': np.mean(cv_results['test_TN']),
                     'test mean FP': np.mean(cv_results['test_FP']),
                     'test mean FN': np.mean(cv_results['test_FN']),
                     'train std TP': np.std(cv_results['train_TP']),
                     'train std TN': np.std(cv_results['train_TN']),
                     'train std FP': np.std(cv_results['train_FP']),
                     'train std FN': np.std(cv_results['train_FN']),
                     'test std TP': np.std(cv_results['test_TP']),
                     'test std TN': np.std(cv_results['test_TN']),
                     'test std FP': np.std(cv_results['test_FP']),
                     'test std FN': np.std(cv_results['test_FN']),
                     'test mean c_MCC': np.mean(cv_results['test_c_MCC']),
                     'test std c_MCC': np.std(cv_results['test_c_MCC']),
                     'train mean c_MCC': np.mean(cv_results['train_c_MCC']),
                     'train std c_MCC': np.std(cv_results['train_c_MCC']),
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

        lr = CSH.UniformFloatHyperparameter('lr', lower=1e-6, upper=1e-1, default_value='1e-2', log=True)
        optimizer = CSH.CategoricalHyperparameter('optimizer', ['Adam', 'SGD', 'Adadelta', 'RMSprop'])
        sgd_momentum = CSH.UniformFloatHyperparameter('sgd_momentum', lower=0.0, upper=0.99, default_value=0.9,
                                                      log=False)
        dropout_rate_1 = CSH.UniformFloatHyperparameter('dropout_rate_1', lower=0.0, upper=0.5, default_value=0.2,
                                                      log=False)
        dropout_rate_2 = CSH.UniformFloatHyperparameter('dropout_rate_2', lower=0.0, upper=0.5, default_value=0.2,
                                                        log=False)
        dropout_rate_3 = CSH.UniformFloatHyperparameter('dropout_rate_3', lower=0.0, upper=0.5, default_value=0.2,
                                                        log=False)
        dropout_rate_4 = CSH.UniformFloatHyperparameter('dropout_rate_4', lower=0.0, upper=0.5, default_value=0.2,
                                                        log=False)
        num_fc_units_1 = CSH.UniformIntegerHyperparameter('num_fc_units_1', lower=512, upper=2048,
                                                          default_value=1024, log=True)
        num_fc_units_2 = CSH.UniformIntegerHyperparameter('num_fc_units_2', lower=256, upper=512, default_value=256,
                                                          log=True)
        num_fc_units_3 = CSH.UniformIntegerHyperparameter('num_fc_units_3', lower=64, upper=256, default_value=128,
                                                          log=True)
        num_fc_units_4 = CSH.UniformIntegerHyperparameter('num_fc_units_4', lower=8, upper=64, default_value=32,
                                                          log=True)
        activation = CSH.CategoricalHyperparameter('activation', ['tanh', 'relu'])

        cs.add_hyperparameters([lr, optimizer, sgd_momentum, dropout_rate_1,
                                dropout_rate_2, dropout_rate_3, dropout_rate_4,
                                num_fc_units_1, num_fc_units_2,
                                num_fc_units_3, num_fc_units_4, activation])

        # The hyperparameter sgd_momentum will be used,if the configuration
        # contains 'SGD' as optimizer.
        cond = CS.EqualsCondition(sgd_momentum, optimizer, 'SGD')
        cs.add_condition(cond)

        return cs


class SVMWorker(Worker):
    """BOHB: ROBUST AND EFFICIENT HYPERPARAMETER OPTIMIZATION AT SCALE
    LinearSVM worker hyperparameter tuning."""

    def __init__(self, **kwargs):
        """Initialize DNN with:
        - scroring method for cross-validation
        - dataset for evaluation"""
        super_kwargs = dict(kwargs)
        del super_kwargs['article']
        del super_kwargs['flavor']
        super().__init__(**super_kwargs)

        self.scoring = {'f1': F1, 'accuracy': ACC, 'TN': TN, 'TP': TP, 'FN': FN, 'FP': FP, 'c_MCC': c_MCC}

        # Loads dataset
        if kwargs['flavor'] == 'desc':
            dataset = echr.binary.get_dataset(article=kwargs['article'], flavors=[echr.Flavor.desc])
        elif kwargs['flavor'] == 'bow':
            dataset = echr.binary.get_dataset(article=kwargs['article'], flavors=[echr.Flavor.bow])
        else:
            dataset = echr.binary.get_dataset(article=kwargs['article'])
        self.X, self.y = dataset.load()  # Load in memory the dataset
        # Make process pipeline
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=config['num_pca']))
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=900))
        std_clf = make_pipeline(StandardScaler())
        self.X = std_clf.fit_transform(self.X.toarray())

    def compute(self, config: object, budget, working_directory, *args, **kwargs):
        """Compute method for each BOHB run evaluation.
        Defines data preprocessing pipeline and keras DNN model for particular run.
        data -> StandardScaler -> (any preprocess technique) -> k-fold_cv -> train_and_test_results
        """

        # Wraps Keras model into sklearn classifier (this is a need for use with cross-validation)
        model = svm.SVC(kernel='linear', C=config['c'])
        # if config['kernel'] == 'linear':
        #     model = svm.SVC(kernel=config['kernel'], C=config['c'])
        # elif config['kernel'] == 'rbf':
        #     model = svm.SVC(kernel=config['kernel'], C=config['c'], gamma=config['gamma'])
        # else:
        #     model = svm.SVC(kernel=config['kernel'], C=config['c'], degree=config['degree'], gamma=config['gamma'])

        # k-fld test, produce estimated score per algorithm based on training and test datasets
        if int(budget) == 555:
            b = 5
        elif int(budget) == 1666:
            b = 7
        else:
            b = 10
        kfold = StratifiedKFold(n_splits=b, random_state=_SEED_)
        cv_results = cross_validate(model, self.X, self.y,
                                    cv=kfold, scoring=self.scoring,
                                    return_train_score=True)

        # Returns information about BOHB loss function and all the computed metrics from DNN
        return ({
            'loss': 1-np.mean(cv_results['test_c_MCC']), # remember: HpBandSter always minimizes!
            'info': {'test mean accuracy': np.mean(cv_results['test_accuracy']),
                     'test std accuracy': np.std(cv_results['test_accuracy']),
                     'train mean accuracy': np.mean(cv_results['train_accuracy']),
                     'train std accuracy': np.std(cv_results['train_accuracy']),
                     'train mean f-1': np.mean(cv_results['train_f1']),
                     'train std f-1': np.std(cv_results['train_f1']),
                     'test mean f-1': np.mean(cv_results['test_f1']),
                     'test std f-1': np.std(cv_results['test_f1']),
                     'train mean TP': np.mean(cv_results['train_TP']),
                     'train mean TN': np.mean(cv_results['train_TN']),
                     'train mean FP': np.mean(cv_results['train_FP']),
                     'train mean FN': np.mean(cv_results['train_FN']),
                     'test mean TP': np.mean(cv_results['test_TP']),
                     'test mean TN': np.mean(cv_results['test_TN']),
                     'test mean FP': np.mean(cv_results['test_FP']),
                     'test mean FN': np.mean(cv_results['test_FN']),
                     'train std TP': np.std(cv_results['train_TP']),
                     'train std TN': np.std(cv_results['train_TN']),
                     'train std FP': np.std(cv_results['train_FP']),
                     'train std FN': np.std(cv_results['train_FN']),
                     'test std TP': np.std(cv_results['test_TP']),
                     'test std TN': np.std(cv_results['test_TN']),
                     'test std FP': np.std(cv_results['test_FP']),
                     'test std FN': np.std(cv_results['test_FN']),
                     'test mean c_MCC': np.mean(cv_results['test_c_MCC']),
                     'test std c_MCC': np.std(cv_results['test_c_MCC']),
                     'train mean c_MCC': np.mean(cv_results['train_c_MCC']),
                     'train std c_MCC': np.std(cv_results['train_c_MCC']),
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

        # kernel = CSH.CategoricalHyperparameter('kernel', ['linear', 'rbf', 'poly'])
        # gamma = CSH.UniformFloatHyperparameter('gamma', lower=0.001, upper=100, default_value=1, log=True)
        c = CSH.UniformFloatHyperparameter('c', lower=0.1, upper=100, default_value=0.1, log=True)
        # degree = CSH.UniformIntegerHyperparameter('degree', lower=2, upper=20, default_value=2, log=True)

        cs.add_hyperparameters([c])

        # cs.add_hyperparameters([kernel, gamma, c, degree])

        # cond = CS.EqualsCondition(degree, kernel, 'poly')
        # cs.add_condition(cond)
        # cond = CS.NotEqualsCondition(gamma, kernel, 'linear')
        # cs.add_condition(cond)
        return cs


class ExtraTreesWorker(Worker):
    """BOHB: ROBUST AND EFFICIENT HYPERPARAMETER OPTIMIZATION AT SCALE
    Randowm Forest worker hyperparameter tuning."""

    def __init__(self, **kwargs):
        """Initialize DNN with:
        - scroring method for cross-validation
        - dataset for evaluation"""
        super_kwargs = dict(kwargs)
        del super_kwargs['article']
        del super_kwargs['flavor']
        super().__init__(**super_kwargs)

        self.scoring = {'f1': F1, 'accuracy': ACC, 'TN': TN, 'TP': TP, 'FN': FN, 'FP': FP, 'c_MCC': c_MCC}

        # Loads dataset
        if kwargs['flavor'] == 'desc':
            dataset = echr.binary.get_dataset(article=kwargs['article'], flavors=[echr.Flavor.desc])
        elif kwargs['flavor'] == 'bow':
            dataset = echr.binary.get_dataset(article=kwargs['article'], flavors=[echr.Flavor.bow])
        else:
            dataset = echr.binary.get_dataset(article=kwargs['article'])
        self.X, self.y = dataset.load()  # Load in memory the dataset
        # Make process pipeline
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=config['num_pca']))
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=900))
        # std_clf = make_pipeline(StandardScaler())
        # self.X = std_clf.fit_transform(self.X.toarray())

    def compute(self, config: object, budget, working_directory, *args, **kwargs):
        """Compute method for each BOHB run evaluation.
        Defines data preprocessing pipeline and keras DNN model for particular run.
        data -> StandardScaler -> (any preprocess technique) -> k-fold_cv-> train_and_test_results
        """

        # Wraps Keras model into sklearn classifier (this is a need for use with cross-validation)
        model = ExtraTreesClassifier(n_estimators=config['n_estimators'], criterion=config['criterion'],
                                     max_depth=config['max_depth'],
                                     random_state=_SEED_)
                                       # min_samples_split=config['min_sample_split'],
                                       # min_samples_leaf=config['min_sample_leaf'],
                                       # max_features=config['max_features'], max_leaf_nodes=config['max_leaf_nodes'],
                                       # min_impurity_decrease=config['min_impur_dist'])

        # k-fld test, produce estimated score per algorithm based on training and test datasets
        if int(budget) == 555:
            b = 5
        elif int(budget) == 1666:
            b = 7
        else:
            b = 10
        kfold = StratifiedKFold(n_splits=b, random_state=_SEED_)
        cv_results = cross_validate(model, self.X, self.y,
                                    cv=kfold, scoring=self.scoring,
                                    return_train_score=True)

        # Returns information about BOHB loss function and all the computed metrics from DNN
        return ({
            'loss': 1-np.mean(cv_results['test_c_MCC']),  # remember: HpBandSter always minimizes!
            'info': {'test mean accuracy': np.mean(cv_results['test_accuracy']),
                     'test std accuracy': np.std(cv_results['test_accuracy']),
                     'train mean accuracy': np.mean(cv_results['train_accuracy']),
                     'train std accuracy': np.std(cv_results['train_accuracy']),
                     'train mean f-1': np.mean(cv_results['train_f1']),
                     'train std f-1': np.std(cv_results['train_f1']),
                     'test mean f-1': np.mean(cv_results['test_f1']),
                     'test std f-1': np.std(cv_results['test_f1']),
                     'train mean TP': np.mean(cv_results['train_TP']),
                     'train mean TN': np.mean(cv_results['train_TN']),
                     'train mean FP': np.mean(cv_results['train_FP']),
                     'train mean FN': np.mean(cv_results['train_FN']),
                     'test mean TP': np.mean(cv_results['test_TP']),
                     'test mean TN': np.mean(cv_results['test_TN']),
                     'test mean FP': np.mean(cv_results['test_FP']),
                     'test mean FN': np.mean(cv_results['test_FN']),
                     'train std TP': np.std(cv_results['train_TP']),
                     'train std TN': np.std(cv_results['train_TN']),
                     'train std FP': np.std(cv_results['train_FP']),
                     'train std FN': np.std(cv_results['train_FN']),
                     'test std TP': np.std(cv_results['test_TP']),
                     'test std TN': np.std(cv_results['test_TN']),
                     'test std FP': np.std(cv_results['test_FP']),
                     'test std FN': np.std(cv_results['test_FN']),
                     'test mean c_MCC': np.mean(cv_results['test_c_MCC']),
                     'test std c_MCC': np.std(cv_results['test_c_MCC']),
                     'train mean c_MCC': np.mean(cv_results['train_c_MCC']),
                     'train std c_MCC': np.std(cv_results['train_c_MCC']),
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
                                                        default_value=10, log=True)
        criterion = CSH.CategoricalHyperparameter('criterion', ['gini', 'entropy'])
        max_depth = CSH.UniformIntegerHyperparameter('max_depth', lower=100, upper=1000,
                                                     default_value=1000, log=True)
        # min_sample_split = CSH.UniformIntegerHyperparameter('min_sample_split', lower=2, upper=100,
        #                                                   default_value=2, log=True)
        # min_sample_leaf = CSH.UniformIntegerHyperparameter('min_sample_leaf', lower=1, upper=100,
        #                                                    default_value=50, log=True)
        # max_features = CSH.CategoricalHyperparameter('max_features', ['auto', 'sqrt', 'log2'])
        # max_leaf_nodes = CSH.UniformIntegerHyperparameter('max_leaf_nodes', lower=10, upper=1000,
        #                                                   default_value=500, log=True)
        # min_impur_dist = CSH.UniformFloatHyperparameter('min_impur_dist', lower=0.1, upper=1.0,
        #                                                 default_value=0.5, log=True)
        cs.add_hyperparameters([n_estimators, criterion, max_depth])

        # cs.add_hyperparameters([n_estimators, criterion, max_depth,
        #                         min_sample_split, min_sample_leaf,
        #                         max_features, max_leaf_nodes, min_impur_dist])

        return cs


class MyKerasClassifier(KerasClassifier):
    """Inherited class from KerasClassifier sklearn wrapper.
    Usage reason:
        During cross-validation, StratifiedKFold splitting cannot operate on one-hot encoded y-vector,
        so after StratifiedKFold, y-vector is passed to KerasClassifier.fit() method which
        should be fitted with one-hot encoded y-vector.
        This is the way how to break into between KFold and Classifier to change the shape of y-vector."""
    def fit(self, x, y, sample_weight=None, **kwargs):
        y = np.array(keras.utils.to_categorical(y, 2))  # Changed original operation
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError('Invalid shape for y: ' + str(y.shape))
        self.n_classes_ = len(self.classes_)
        if sample_weight is not None:
            kwargs['sample_weight'] = sample_weight
        return super(KerasClassifier, self).fit(x, y, **kwargs)

