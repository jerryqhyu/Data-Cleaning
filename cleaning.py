from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
import autograd.numpy as np
from tree import NoisyLabeler
from autograd.optimizers import adam
from autograd.scipy.misc import logsumexp
from autograd import grad
from util import data_reader
install_aliases()


class Cleaner():

    def __init__(self, theta):
        reader = data_reader()
        # saver = image_saver()
        n, a, b, c, d = reader.load_mnist()

        self.train_data = np.round(a)[:10000]
        self.valid_data = np.round(c)[:5000]
        self.train_labels = b[:10000]
        self.valid_labels = d[:5000]

        self.w = np.zeros((self.train_labels.shape[1], self.train_data.shape[1]))
        self.theta = theta

    def logistic_likelihood(self, x, w):
        log_likelihood = logistic_ll(x, w)
        return np.exp(log_likelihood)

    def weighted_likelihood(self, x, y, w, theta, likelihood_fcn):
        likelihood = likelihood_fcn(x, w)
        return np.sum(np.multiply(y, np.log(np.dot(theta, likelihood.T)).T))

    def train_logistic(self, learning_rate=0.01, epoch=500):
        gradient = grad(self.weighted_likelihood, argnum=2)
        for i in range(epoch):
            self.w += gradient(self.train_data, self.train_labels, self.w, self.theta, self.logistic_likelihood)
        return self.w

    def metrics(self):
        # likelihood
        ll = np.multiply(self.train_labels, pred_ll(self.train_data, self.w))
        test_ll = np.multiply(self.valid_labels, pred_ll(self.valid_data, self.w))

        # accuracy
        pred = pred_ll(self.train_data, self.w)
        test_pred = pred_ll(self.valid_data, self.w)
        print("average predictive log ll for training set is:")
        print(np.sum(ll)/30)
        print("average predictive log ll for test set is:")
        print(np.sum(test_ll)/10000)

        print("average predictive accuracy for training set is:")
        print(predictive_accuracy(pred, self.train_labels))
        print("average predictive accuracy for test set is:")
        print(predictive_accuracy(test_pred, self.valid_labels))


def logistic_ll(x, w):
    return np.dot(x, w.T) - np.tile(logsumexp(np.dot(x, w.T), axis=1), 10).reshape(10, x.shape[0]).T


def fit_logistic(images, labels):
    x = images
    y = labels
    w = np.zeros((labels.shape[1], images.shape[1]))
    gradient = grad(logit_ll, argnum=2)
    for i in range(5000):
        w += gradient(x, y, w) * 0.001
    return w


def pred_ll(x, w):
    return logll(np.dot(x, w.T))


def logll(K):
    return K - np.tile(logsumexp(K, axis=1), 10).reshape(10, K.shape[0]).T


def predictive_accuracy(proposed, true):
    return np.sum(np.equal(np.argmax(proposed, axis=1), np.argmax(true, axis=1)))/true.shape[0] * 100


if __name__ == '__main__':
    cleaner = Cleaner(0)
    cleaner.train_logistic()
    cleaner.metrics()
