import numpy as np
import pytest

from tempo.kfserving.k8s import KFServingKubernetesRuntime
from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model


@pytest.mark.skip(reason="needs k8s cluster")
def test_kfserving():
    model = Model(
        name="sklearn-iris2",
        runtime=KFServingKubernetesRuntime(),
        platform=ModelFramework.SKLearn,
        uri="gs://kfserving-samples/models/sklearn/iris",
        local_folder="sklearn/model",
    )
    # model.deploy()
    res = model(np.array([[1, 2, 3, 4]]))
    print(res)
