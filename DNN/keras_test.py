try:
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras import backend as K
except:
    raise ImportError("For this example you need to install keras.")

# Usage of https://github.com/aquemy/ECHR-OD_loader from different directory
import sys
sys.path.insert(0, 'C:\\Users\\Amadeusz\\Documents\\GitHub\\ECHR-OD_loader\\')
sys.path.insert(0, 'C:\\Users\\Amadeusz\\Documents\\GitHub\\ECHR-OD_loader\\echr\\')
import echr

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np
import pprint

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

from hpbandster.core.worker import Worker

import logging
logging.basicConfig(level=logging.DEBUG)

_SHAPE_ = None

class KerasWorker(Worker):
    """BOHB: ROBUST AND EFFICIENT HYPERPARAMETER OPTIMIZATION AT SCALE
    Keras framework class for DNN hyperparameter tuning."""

    def __init__(self, **kwargs):
        """Initialize DNN with:
        - batch size for each worker,
        - number of classes to predict,
        - scroring method for cross-validation
        - dataset for evaluation"""

        super().__init__(**kwargs)

        self.batch_size = 64
        self.num_classes = 2
        self.scoring = ['f1', 'accuracy']

        # Loads dataset
        dataset = echr.binary.get_dataset(article='11', flavors=[echr.Flavor.desc])  # Define the dataset model
        self.X, self.y = dataset.load()  # Load in memory the dataset
        # Make process pipeline
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=config['num_pca']))
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=900))
        std_clf = make_pipeline(StandardScaler())
        self.X = std_clf.fit_transform(self.X.toarray())
        global _SHAPE_
        _SHAPE_ = self.X.shape[1]

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
        kfold = StratifiedKFold(n_splits=10, random_state=7)
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
        global _SHAPE_
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
        num_fc_units_1 = CSH.UniformIntegerHyperparameter('num_fc_units_1', lower=512, upper=_SHAPE_,
                                                          default_value=_SHAPE_, log=True)
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


# This part is only used to validation of above code in one run.
# keras_optimize.py implements whole BOHB searching space procedure
if __name__ == "__main__":
    worker = KerasWorker(run_id='0')
    cs = worker.get_configspace()

    config = cs.sample_configuration().get_dictionary()
    print(config)
    res = worker.compute(config=config, budget=2, working_directory='.')
    pp = pprint.PrettyPrinter()
    pp.pprint(res)
