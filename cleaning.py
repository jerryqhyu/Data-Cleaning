from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
import autograd.numpy as np
from util import image_saver
from cleaner import Cleaner
install_aliases()


def execute_net(mode='eye', mat=np.eye(10), learning_rate=0.01, epoch=100, power_level=1):
    if mode is 'eye':
        cleaner = Cleaner(np.eye(10), power_level=power_level)
    elif mode is 'pre':
        cleaner = Cleaner(mat, power_level=power_level)
    else:
        cleaner = Cleaner(np.eye(10), power_level=power_level, train=True)
    cleaner.train_net(learning_rate=learning_rate, epoch=epoch)
    cleaner.net_metrics()


def execute_logistic(mode='eye', mat=np.eye(10), learning_rate=0.01, epoch=100, power_level=1):
    if mode is 'eye':
        cleaner = Cleaner(np.eye(10), power_level=power_level)
    elif mode is 'pre':
        cleaner = Cleaner(mat, power_level=power_level)
    else:
        cleaner = Cleaner(np.eye(10), power_level=power_level, train=True)
    cleaner.train_logistic(learning_rate=learning_rate, epoch=epoch)
    cleaner.metrics()
    saver = image_saver()
    saver.save_images(cleaner.w, 'theta'+str(power_level)+mode)


if __name__ == '__main__':
    mat = np.eye(10) * 0.9 + 0.01
    # execute_net(mode='eye', mat=np.eye(10), learning_rate=0.01, epoch=100, power_level=1)
    execute_logistic(mode='eye', mat=mat, learning_rate=0.005, epoch=500, power_level=1)
