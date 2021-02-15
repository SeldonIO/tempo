from tempo.kfserving.k8s import KFServingKubernetesRuntime
from tempo.serve.model import Model
from tempo.serve.metadata import ModelFramework
import numpy as np

def test_kfserving():
    model = Model(
        name="sklearn-iris2",
        runtime=KFServingKubernetesRuntime(),
        platform=ModelFramework.SKLearn,
        uri="gs://kfserving-samples/models/sklearn/iris",
        local_folder="sklearn/model"
    )
    #model.deploy()
    res = model(np.array([[1, 2, 3, 4]]))
    print(res)