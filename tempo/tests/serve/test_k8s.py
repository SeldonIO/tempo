import pytest
import numpy as np

from kubernetes import client

from tempo.serve.model import Model
from tempo.seldon.k8s import SeldonKubernetesRuntime


@pytest.mark.skip(reason="needs k8s cluster")
def test_deploy_k8s(k8s_sklearn_model: Model, k8s_runtime: SeldonKubernetesRuntime):
    crd_api = client.CustomObjectsApi()
    sdep = crd_api.get_namespaced_custom_object(
        "machinelearning.seldon.io",
        "v1",
        k8s_runtime.k8s_options.namespace,
        "seldondeployments",
        k8s_sklearn_model.details.name,
    )

    assert sdep["status"]["state"] == "Available"


@pytest.mark.skip(reason="needs k8s cluster")
@pytest.mark.parametrize(
    "x_input",
    [[[1, 2, 3, 4]], np.array([[1, 2, 3, 4]]), {"data": {"ndarray": [[1, 2, 3, 4]]}}],
)
def test_sklearn_k8s(k8s_sklearn_model: Model, x_input):
    y_pred = k8s_sklearn_model(x_input)

    np.testing.assert_allclose(y_pred, [[0, 0, 0.99]], atol=1e-2)


#  def test_undeploy_k8s(k8s_sklearn_model: Model, docker_runtime: SeldonDockerRuntime):
#  sklearn_model.deploy()
#  time.sleep(2)

#  sklearn_model.undeploy()

#  with pytest.raises(docker.errors.NotFound):
#  docker_runtime._get_container(sklearn_model._details)
