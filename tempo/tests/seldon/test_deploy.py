from tempo.serve.metadata import ModelFramework, KubernetesOptions
from tempo.serve.model import Model
from tempo.seldon.deploy import SeldonDeployRuntime
from tempo.seldon.k8s import SeldonKubernetesRuntime
import os
import pytest
import numpy as np

@pytest.mark.skip("needs deploy cluster")
def test_deploy():
    rt = SeldonDeployRuntime(
        host="http://34.105.136.157/seldon-deploy/api/v1alpha1",
        user="admin@kubeflow.org",
        password="12341234",
        k8s_options=KubernetesOptions(namespace="seldon"),
    )

    sklearn_model = Model(
        name="test-iris-sklearn",
        runtime=rt,
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris",
        local_folder=os.getcwd() + "/sklearn",
    )

    sklearn_model.deploy()
    sklearn_model.wait_ready()
    sklearn_model(np.array([[4.9, 3.1, 1.5, 0.2]]))

@pytest.mark.skip("needs deploy cluster")
def test_deploy_yaml():
    rt = SeldonDeployRuntime(
        host="http://34.105.136.157/seldon-deploy/api/v1alpha1",
        user="admin@kubeflow.org",
        password="12341234",
        k8s_options=KubernetesOptions(namespace="seldon"),
    )

    sklearn_model = Model(
        name="test-iris-sklearn",
        runtime=rt,
        platform=ModelFramework.SKLearn,
        uri="gs://seldon-models/sklearn/iris",
        local_folder=os.getcwd() + "/sklearn",
    )

    srt = SeldonKubernetesRuntime(k8s_options=KubernetesOptions(namespace="seldon"))
    sklearn_model.set_runtime(srt)
    expected = sklearn_model.to_k8s_yaml()
    sklearn_model.set_runtime(rt)
    assert sklearn_model.to_k8s_yaml() == expected
