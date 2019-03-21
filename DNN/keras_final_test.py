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
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import StratifiedKFold, cross_validate
import numpy as np


class Check_DNN():

    def __init__(self):
        self.batch_size = 64
        self.num_classes = 2
        self.epochs = 16
        self.scoring = ['f1', 'accuracy']
        self.config = {'sgd_momentum': 0.18890424476805087,
                       'num_fc_units_2': 51,
                       'num_fc_units_3': 110,
                       'optimizer': 'SGD',
                       'num_fc_units_1': 231,
                       'dropout_rate': 0.15195434176307368,
                       'lr': 2.0072660857221496e-05}

        # Loads dataset
        dataset = echr.binary.get_dataset(article='11', flavors=[echr.Flavor.desc])  # Define the dataset model
        self.X, self.y = dataset.load()  # Load in memory the dataset
        # Make process pipeline
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=config['num_pca']))
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=900))
        std_clf = make_pipeline(StandardScaler())
        self.X = std_clf.fit_transform(self.X.toarray())
        self.model = self.MyKerasClassifier(build_fn=self.keras_model,
                                            epochs=self.epochs,
                                            batch_size=self.batch_size,
                                            verbose=0)
        # k-fld test, produce estimated score per algorithm based on training and test datasets
        kfold = StratifiedKFold(n_splits=10, random_state=7)
        self.cv_results = cross_validate(self.model, self.X, self.y,
                                         cv=kfold, scoring=self.scoring,
                                         return_train_score=True)
        print('Test mean accuracy : {} std{}'.format(np.mean(self.cv_results['test_accuracy']),
                                                             np.std(self.cv_results['test_accuracy'])))
        print('Test mean f-1 : {} std {}'.format(np.mean(self.cv_results['test_f1']),
                                                        np.std(self.cv_results['test_f1'])))
        print('Train mean accuracy : {} std {}'.format(np.mean(self.cv_results['train_accuracy']),
                                                              np.std(self.cv_results['train_accuracy'])))
        print('Train mean f-1 : {} std {}'.format(np.mean(self.cv_results['train_f1']),
                                                         np.std(self.cv_results['train_f1'])))

    def keras_model(self):
        model = Sequential()
        model.add(Dense(self.config['num_fc_units_1'], input_dim=self.X.shape[1], activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        model.add(Dense(self.config['num_fc_units_2'], activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        model.add(Dense(self.config['num_fc_units_3'], activation='relu'))
        model.add(Dropout(self.config['dropout_rate']))
        model.add(Dense(2, activation='softmax'))

        # Choose an optimizer to use
        if self.config['optimizer'] == 'Adam':
            optimizer = keras.optimizers.Adam(lr=self.config['lr'])
        else:
            optimizer = keras.optimizers.SGD(lr=self.config['lr'], momentum=self.config['sgd_momentum'])

        # Compile the model with binary crossentropy as a loss function and binary accuracy as a metric
        model.compile(loss=keras.losses.binary_crossentropy,
                      optimizer=optimizer,
                      metrics=['binary_accuracy'])
        return model

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


if __name__ == "__main__":
    DNN = Check_DNN()