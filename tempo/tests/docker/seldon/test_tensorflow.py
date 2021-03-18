import pytest


def test_launch_tensorflow(tfserving_cifar10_resenet32_model):
    tfserving_cifar10_resenet32_model.deploy()
    tfserving_cifar10_resenet32_model.wait_ready()
