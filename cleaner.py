from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
import autograd.numpy as np
from labeler import NoisyLabeler
from autograd.scipy.misc import logsumexp
from autograd import grad
from util import data_reader, image_saver
install_aliases()


class Cleaner():

    def __init__(self, theta, power_level=1, train=False):
        reader = data_reader()
        n, a, b, c, d = reader.load_mnist()

        noisy = NoisyLabeler(a, b, c, d, power_level=power_level)
        noisy.power_level()
        a, bad, c, dad = noisy.get_noisy_train_valid()

        self.true_train = b[:25000]
        self.true_valid = d[:5000]

        self.train_data = np.round(a)[:25000]
        self.valid_data = np.round(c)[:5000]
        self.train_labels = bad[:25000]
        self.valid_labels = dad[:5000]

        self.w = np.zeros(
            (self.train_labels.shape[1], self.train_data.shape[1]))
        self.theta = theta
        self.layer_1 = np.random.randn(784, 256)
        self.bias_1 = np.random.randn(256)
        self.layer_2 = np.random.randn(256, 10)

        self.train = train

    def logistic_likelihood(self, x, w):
        log_likelihood = logistic_ll(x, w)
        return np.exp(log_likelihood)

    def weighted_likelihood(self, x, y, w, theta, likelihood_fcn):
        likelihood = likelihood_fcn(x, w)
        return np.sum(np.multiply(y, np.dot(theta, likelihood.T).T))

    def net_likelihood(self, l1, b1, l2, x, y, theta):
        first = np.maximum(0, (np.dot(l1.T, x.T).T + b1).T)
        out = softmax(np.dot(l2.T, first).T)
        return np.sum(np.multiply(y, np.dot(theta, out.T).T))

    def train_logistic(self, learning_rate=0.01, epoch=500):
        gradient = grad(self.weighted_likelihood, argnum=2)
        gradient_theta = grad(self.weighted_likelihood, argnum=3)
        if self.train:
            for i in range(epoch):
                print("This is iteration {}.".format(i))
                self.w += gradient(self.train_data, self.train_labels, self.w,
                                   self.theta, self.logistic_likelihood) * learning_rate
            for i in range(500):
                print("This is iteration {} optimizing theta.".format(i))
                self.theta += gradient_theta(self.train_data, self.train_labels,
                                             self.w, self.theta, self.logistic_likelihood) * 0.0001
                # normalize theta
                self.theta = (self.theta.T / np.sum(self.theta, axis=1)).T

        for i in range(epoch):
            print("This is iteration {}.".format(i))
            g = gradient(self.train_data, self.train_labels, self.w,
                         self.theta, self.logistic_likelihood)
            self.w += g * learning_rate
        print(self.theta)

        saver = image_saver()
        rxw = self.logistic_likelihood(self.train_data, self.w)
        cxw = np.dot(self.theta, self.logistic_likelihood(self.train_data, self.w).T).T
        indices = np.where(np.equal(np.argmax(rxw, axis=1), np.argmax(self.true_train, axis=1))==False)[0][-30:]
        saver.save_images(self.train_data[indices], 'mispredicted')
        return self.w

    def train_net(self, learning_rate=0.01, epoch=500):
        print("=== Training Neural Net ===")
        gradient_l1 = grad(self.net_likelihood, argnum=0)
        gradient_b1 = grad(self.net_likelihood, argnum=1)
        gradient_l2 = grad(self.net_likelihood, argnum=2)
        gradient_theta = grad(self.net_likelihood, argnum=5)
        if self.train:
            for i in range(epoch):
                print("This is iteration {}.".format(i + 1))
                gl1 = gradient_l1(self.layer_1, self.bias_1, self.layer_2, self.train_data, self.train_labels, self.theta)
                gb1 = gradient_b1(self.layer_1, self.bias_1, self.layer_2, self.train_data, self.train_labels, self.theta)
                gl2 = gradient_l2(self.layer_1, self.bias_1, self.layer_2, self.train_data, self.train_labels, self.theta)
                self.layer_1 += learning_rate * gl1
                self.bias_1 += learning_rate * gb1
                self.layer_2 += learning_rate * gl2

            for i in range(15):
                print("This is iteration {} optimizing theta.".format(i))
                self.theta += gradient_theta(self.layer_1, self.bias_1, self.layer_2, self.train_data, self.train_labels, self.theta) * 0.0001
                # normalize theta
                self.theta = (self.theta.T / np.sum(self.theta, axis=1)).T

        for i in range(epoch):
            print("This is iteration {}.".format(i + 1))
            gl1 = gradient_l1(self.layer_1, self.bias_1, self.layer_2, self.train_data, self.train_labels, self.theta)
            gb1 = gradient_b1(self.layer_1, self.bias_1, self.layer_2, self.train_data, self.train_labels, self.theta)
            gl2 = gradient_l2(self.layer_1, self.bias_1, self.layer_2, self.train_data, self.train_labels, self.theta)
            self.layer_1 += learning_rate * gl1
            self.bias_1 += learning_rate * gb1
            self.layer_2 += learning_rate * gl2
        print(self.theta)

    def metrics(self):
        # accuracy
        pred = pred_ll(self.train_data, self.w)
        test_pred = pred_ll(self.valid_data, self.w)
        print("average predictive accuracy for training set is:")
        print(predictive_accuracy(pred, self.true_train))
        print("average predictive accuracy for test set is:")
        print(predictive_accuracy(test_pred, self.true_valid))

    def net_metrics(self):
        # accuracy
        pred = net_ll(self.layer_1, self.bias_1, self.layer_2,
                      self.train_data)
        test_pred = net_ll(self.layer_1, self.bias_1, self.layer_2,
                      self.valid_data)
        print("average predictive accuracy for training set is:")
        print(predictive_accuracy(pred, self.true_train))
        print("average predictive accuracy for test set is:")
        print(predictive_accuracy(test_pred, self.true_valid))


def logistic_ll(x, w):
    return np.dot(x, w.T) - np.tile(logsumexp(np.dot(x, w.T), axis=1), 10).reshape(10, x.shape[0]).T


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.repeat(np.max(x, axis=1), (10)).reshape(x.shape))
    return (e_x.T / np.sum(e_x, axis=1)).T


def pred_ll(x, w):
    return logll(np.dot(x, w.T))


def logll(K):
    return K - np.tile(logsumexp(K, axis=1), 10).reshape(10, K.shape[0]).T


def net_ll(l1, b1, l2, x):
    first = np.maximum(0, (np.dot(l1.T, x.T).T + b1).T)
    out = softmax(np.dot(l2.T, first).T)
    return out


def predictive_accuracy(proposed, true):
    return np.sum(np.equal(np.argmax(proposed, axis=1), np.argmax(true, axis=1))) / true.shape[0] * 100
