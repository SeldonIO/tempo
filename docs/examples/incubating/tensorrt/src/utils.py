import matplotlib.pyplot as plt
import numpy as np
from alibi_detect.utils.perturbation import apply_mask
from src.data import Cifar10


def show_image(X):
    plt.imshow(X.reshape(32, 32, 3))
    plt.axis("off")
    plt.show()


def create_cifar10_outlier(data: Cifar10):
    idx = 1
    X = data.X_train[idx : idx + 1]
    np.random.seed(0)
    X_mask, mask = apply_mask(
        X.reshape(1, 32, 32, 3),
        mask_size=(10, 10),
        n_masks=1,
        channels=[0, 1, 2],
        mask_type="normal",
        noise_distr=(0, 1),
        clip_rng=(0, 1),
    )
    return X_mask
