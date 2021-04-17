import numpy as np
from alibi.datasets import fetch_adult


class AdultData(object):
    def __init__(
        self,
    ):
        adult = fetch_adult()
        data = adult.data
        target = adult.target
        self.feature_names = adult.feature_names
        self.category_map = adult.category_map
        np.random.seed(0)
        data_perm = np.random.permutation(np.c_[data, target])
        data = data_perm[:, :-1]
        target = data_perm[:, -1]
        idx = 30000
        self.X_train, self.Y_train = data[:idx, :], target[:idx]
        self.X_test, self.Y_test = data[idx + 1 :, :], target[idx + 1 :]
