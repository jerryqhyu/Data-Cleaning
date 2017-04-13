from __future__ import absolute_import
from __future__ import print_function
from future.standard_library import install_aliases
import autograd.numpy as np
from util import image_saver
from cleaner import Cleaner
install_aliases()


def execute_net(mode='eye', mat=np.eye(10), learning_rate=0.01, epoch=100, power_level=1):
    if mode is 'eye':
        cleaner = Cleaner(np.eye(10))
    elif mode is 'pre':
        cleaner = Cleaner(mat)
    else:
        cleaner = Cleaner(np.eye(10), train=True)
    cleaner.train_net(learning_rate=learning_rate, epoch=epoch)
    cleaner.net_metrics()


def execute_logistic(mode='eye', mat=np.eye(10), learning_rate=0.01, epoch=100, power_level=1):
    if mode is 'eye':
        cleaner = Cleaner(np.eye(10))
    elif mode is 'pre':
        cleaner = Cleaner(mat)
    else:
        cleaner = Cleaner(np.eye(10), train=True)
    cleaner.train_logistic(learning_rate=learning_rate, epoch=epoch)
    cleaner.metrics()
    saver = image_saver()
    saver.save_images(cleaner.w, 'theta')


if __name__ == '__main__':
    # execute_net(mode='eye', mat=np.eye(10), learning_rate=0.01, epoch=100, power_level=1)
    execute_logistic(mode='eye', mat=np.eye(10), learning_rate=0.01, epoch=100, power_level=1)
