# Usage of https://github.com/aquemy/ECHR-OD_loader from different directory
import sys
sys.path.insert(0, 'C:\\Users\\Amadeusz\\Documents\\GitHub\\ECHR-OD_loader\\')
sys.path.insert(0, 'C:\\Users\\Amadeusz\\Documents\\GitHub\\ECHR-OD_loader\\echr\\')
import echr
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import matthews_corrcoef, make_scorer, accuracy_score, f1_score
import numpy as np

_SEED_ = 7
MCC = make_scorer(matthews_corrcoef)
ACC = make_scorer(accuracy_score)
F1 = make_scorer(f1_score)


class CheckXGBoost():
    def __init__(self):
        self.scoring = {'f1': F1, 'accuracy': ACC, 'MCC': MCC}
        # Best Configuration flavor: desc
        # Test mean accuracy: 0.9292207792207792 std 0.04380697852058865
        # Test mean f - 1: 0.959558476926898 std 0.024117773683387442
        # Train mean accuracy: 1.0 std 0.0
        # Train  mean f - 1: 1.0 std 0.0
        # Train  mean MCC: 1.0 std  0.0
        # Test mean MCC: 0.6668310732823477 std 0.28352974046333224
        self.config = {'max_depth': 196, 'lr': 0.024972501951425863, 'n_estimators': 902}

        # Loads dataset
        dataset = echr.binary.get_dataset(article='11', flavors=[echr.Flavor.desc])  # Define the dataset model
        self.X, self.y = dataset.load()  # Load in memory the dataset
        # Make process pipeline
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=config['num_pca']))
        # std_clf = make_pipeline(StandardScaler(), PCA(n_components=900))
        std_clf = make_pipeline(StandardScaler())
        self.X = std_clf.fit_transform(self.X.toarray())
        self.model = XGBClassifier(max_depth=self.config['max_depth'],
                                   learning_rate=self.config['lr'],
                                   n_estimators=self.config['n_estimators'],
                                   n_jobs=4,
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
    # DNN = CheckXGBoost()
    print(matthews_corrcoef([1,1,0], [1,1,0]))