from sklearn import tree
import autograd.numpy as np
from sklearn.metrics import accuracy_score


class NoisyLabeler():

    def __init__(self, train_data, train_labels, valid_data, valid_labels):

        print('=== Loading and Training the Labeler ===')
        self.train_data = train_data
        self.train_labels = np.argmax(train_labels, axis=1)
        self.valid_data = valid_data
        self.valid_labels = np.argmax(valid_labels, axis=1)
        self.clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=8, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_split=1e-07, class_weight=None, presort=False)
        self.clf = self.clf.fit(self.train_data, self.train_labels)

        print('=== Labeling Using the Labeler ===')
        self.predicted_train = self.clf.predict(train_data)
        self.predicted_valid = self.clf.predict(valid_data)

    def get_noisy_train_valid(self):
        n_values = np.max(self.predicted_train) + 1
        one_hot_train = np.eye(n_values)[self.predicted_train]
        one_hot_valid = np.eye(n_values)[self.predicted_valid]
        return one_hot_train, one_hot_valid

    def power_level(self):
        print('The predictive accuracy for training is {}.'.format(accuracy_score(
            self.train_labels, self.predicted_train)))
        print('The predictive accuracy for valid is {}.'.format(accuracy_score(
            self.valid_labels, self.predicted_valid)))
