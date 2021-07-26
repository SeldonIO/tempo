import numpy as np
import pytest
from kubernetes import client

from tempo.seldon.k8s import SeldonKubernetesRuntime
from tempo.serve.metadata import KubernetesRuntimeOptions


def test_create_k8s_runtime():
    rto = KubernetesRuntimeOptions()
    rt = SeldonKubernetesRuntime(rto)
    assert rt.runtime_options.runtime == "tempo.seldon.SeldonKubernetesRuntime"


@pytest.mark.skip(reason="needs k8s cluster")
def test_deploy_k8s(sklearn_model, runtime: SeldonKubernetesRuntime):
    crd_api = client.CustomObjectsApi()
    sdep = crd_api.get_namespaced_custom_object(
        "machinelearning.seldon.io",
        "v1",
        runtime.namespace,
        "seldondeployments",
        sklearn_model.details.name,
    )

    assert sdep["status"]["state"] == "Available"


@pytest.mark.skip(reason="needs k8s cluster")
@pytest.mark.parametrize(
    "x_input",
    [[[1, 2, 3, 4]], np.array([[1, 2, 3, 4]]), {"data": {"ndarray": [[1, 2, 3, 4]]}}],
)
def test_sklearn_k8s(sklearn_model_deployed, x_input):
    y_pred = sklearn_model_deployed(x_input)

    np.testing.assert_allclose(y_pred, [[0, 0, 0.99]], atol=1e-2)


@pytest.mark.skip(reason="needs k8s cluster")
@pytest.mark.parametrize("x_input", [np.array([[1, 2, 3, 4]])])
def test_pipeline_k8s(inference_pipeline_deployed, x_input):
    y_pred = inference_pipeline_deployed.predict(payload=x_input)

    np.testing.assert_allclose(y_pred, [2.0], atol=1e-2)
