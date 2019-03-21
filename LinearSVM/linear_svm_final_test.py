# Usage of https://github.com/aquemy/ECHR-OD_loader from different directory
import sys
sys.path.insert(0, 'C:\\Users\\Amadeusz\\Documents\\GitHub\\ECHR-OD_loader\\')
sys.path.insert(0, 'C:\\Users\\Amadeusz\\Documents\\GitHub\\ECHR-OD_loader\\echr\\')
import echr
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import matthews_corrcoef, make_scorer, accuracy_score, f1_score
import numpy as np

_SEED_ = 7
MCC = make_scorer(matthews_corrcoef)
ACC = make_scorer(accuracy_score)
F1 = make_scorer(f1_score)


class LinearSVMWorker():
    def __init__(self):
        self.scoring = {'f1': F1, 'accuracy': ACC, 'MCC': MCC}
        # Best Configuration flavor desc
        # Test mean accuracy : 0.864069264069264 std 0.03128885021574484
        # Test mean MCC: 0.1766265508260269 std 0.27166718893399866
        self.config = {'c': 1}

        # Loads dataset
        dataset = echr.binary.get_dataset(article='1', flavors=[echr.Flavor.desc])  # Define the dataset model
        self.X, self.y = dataset.load()  # Load in memory the dataset
        # Make process pipeline
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=config['num_pca']))
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=900))
        std_clf = make_pipeline(StandardScaler())
        self.X = std_clf.fit_transform(self.X.toarray())
        self.model = svm.SVC(kernel='linear',
                             C=self.config['c'],
                             random_state=_SEED_)
        # k-fld test, produce estimated score per algorithm based on training and test datasets
        kfold = StratifiedKFold(n_splits=10, random_state=_SEED_)
        self.cv_results = cross_validate(self.model, self.X, self.y,
                                         cv=kfold, scoring=self.scoring,
                                         return_train_score=True)
        print('Test mean accuracy : {} std {}'.format(np.mean(self.cv_results['test_accuracy']),
                                                             np.std(self.cv_results['test_accuracy'])))
        print('Test mean f-1 : {} std {}'.format(np.mean(self.cv_results['test_f1']),
                                                        np.std(self.cv_results['test_f1'])))
        print('Train mean accuracy : {} std {}'.format(np.mean(self.cv_results['train_accuracy']),
                                                              np.std(self.cv_results['train_accuracy'])))
        print('Train mean f-1 : {} std {}'.format(np.mean(self.cv_results['train_f1']),
                                                         np.std(self.cv_results['train_f1'])))
        print('Train mean MCC : {} std {}'.format(np.mean(self.cv_results['train_MCC']),
                                                       np.std(self.cv_results['train_MCC'])))
        print('Test mean MCC : {} std {}'.format(np.mean(self.cv_results['test_MCC']),
                                                  np.std(self.cv_results['test_MCC'])))


if __name__ == "__main__":
    DNN = LinearSVMWorker()