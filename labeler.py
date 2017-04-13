from sklearn.linear_model import LogisticRegression
import autograd.numpy as np
from sklearn.metrics import accuracy_score


class NoisyLabeler():

    def __init__(self, train_data, train_labels, valid_data, valid_labels, power_level=7):

        print('=== Loading and Training the Labeler ===')
        self.train_data = train_data
        self.train_labels = np.argmax(train_labels, axis=1)
        self.valid_data = valid_data
        self.valid_labels = np.argmax(valid_labels, axis=1)

        self.clf = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None,
                                      random_state=None, solver='liblinear', max_iter=power_level, multi_class='ovr', verbose=0, warm_start=False, n_jobs=1)

        self.clf = self.clf.fit(self.train_data, self.train_labels)

        print('=== Labeling Using the Labeler ===')
        self.predicted_train = self.clf.predict(train_data)
        self.predicted_valid = self.clf.predict(valid_data)

    def get_noisy_train_valid(self):
        n_values = np.max(self.predicted_train) + 1
        one_hot_train = np.eye(n_values)[self.predicted_train]
        one_hot_valid = np.eye(n_values)[self.predicted_valid]
        return self.train_data, one_hot_train, self.valid_data, one_hot_valid

    def power_level(self):
        print('The predictive accuracy for training is {}.'.format(accuracy_score(
            self.train_labels, self.predicted_train)))
        print('The predictive accuracy for valid is {}.'.format(accuracy_score(
            self.valid_labels, self.predicted_valid)))
