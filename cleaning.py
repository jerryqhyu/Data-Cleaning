from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
import autograd.numpy as np
from tree import NoisyLabeler
from autograd.scipy.misc import logsumexp
from autograd import grad
from util import data_reader, image_saver
install_aliases()


class Cleaner():

    def __init__(self, theta):
        reader = data_reader()
        n, a, b, c, d = reader.load_mnist()

        noisy = NoisyLabeler(a, b, c, d)
        noisy.power_level()
        a, bad, c, dad = noisy.get_noisy_train_valid()

        self.true_train = b[:10000]
        self.true_valid = d[:5000]

        self.train_data = np.round(a)[:10000]
        self.valid_data = np.round(c)[:5000]
        self.train_labels = bad[:10000]
        self.valid_labels = dad[:5000]

        self.w = np.zeros((self.train_labels.shape[1], self.train_data.shape[1]))
        self.theta = theta

    def logistic_likelihood(self, x, w):
        log_likelihood = logistic_ll(x, w)
        return np.exp(log_likelihood)

    def weighted_likelihood(self, x, y, w, theta, likelihood_fcn):
        likelihood = likelihood_fcn(x, w)
        return np.sum(np.multiply(y, np.dot(theta, likelihood.T).T))

    def train_logistic(self, learning_rate=0.01, epoch=500):
        gradient = grad(self.weighted_likelihood, argnum=2)
        gradient_theta = grad(self.weighted_likelihood, argnum=3)
        for i in range(epoch):
            print("This is iteration {}.".format(i))
            self.w += gradient(self.train_data, self.train_labels, self.w, self.theta, self.logistic_likelihood) * learning_rate
        for i in range(100):
            print("This is iteration {} optimizing theta.".format(i))
            self.theta += gradient_theta(self.train_data, self.train_labels, self.w, self.theta, self.logistic_likelihood) * 0.0001
        print(self.theta)
        return self.w

    def metrics(self):
        # likelihood
        ll = np.multiply(self.true_train, pred_ll(self.train_data, self.w))
        test_ll = np.multiply(self.true_valid, pred_ll(self.valid_data, self.w))

        # accuracy
        pred = pred_ll(self.train_data, self.w)
        test_pred = pred_ll(self.valid_data, self.w)
        print("average predictive log ll for training set is:")
        print(np.sum(ll)/10000)
        print("average predictive log ll for test set is:")
        print(np.sum(test_ll)/5000)

        print("average predictive accuracy for training set is:")
        print(predictive_accuracy(pred, self.true_train))
        print("average predictive accuracy for test set is:")
        print(predictive_accuracy(test_pred, self.true_valid))


def logistic_ll(x, w):
    return np.dot(x, w.T) - np.tile(logsumexp(np.dot(x, w.T), axis=1), 10).reshape(10, x.shape[0]).T


def pred_ll(x, w):
    return logll(np.dot(x, w.T))


def logll(K):
    return K - np.tile(logsumexp(K, axis=1), 10).reshape(10, K.shape[0]).T


def predictive_accuracy(proposed, true):
    return np.sum(np.equal(np.argmax(proposed, axis=1), np.argmax(true, axis=1)))/true.shape[0] * 100


if __name__ == '__main__':
    # cleaner = Cleaner(np.eye(10))
    cleaner = Cleaner(np.eye(10) * 0.8 + 0.02)
    cleaner.train_logistic(learning_rate=0.001, epoch=500)
    cleaner.metrics()
    saver = image_saver()
    saver.save_images(cleaner.w, 'theta')
