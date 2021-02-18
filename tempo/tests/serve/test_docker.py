import time
import pytest
import docker
import numpy as np

from tempo.serve.model import Model
from tempo.serve.pipeline import Pipeline
from tempo.seldon.docker import SeldonDockerRuntime


def test_deploy_docker(sklearn_model: Model, docker_runtime: SeldonDockerRuntime):
    sklearn_model.deploy()
    time.sleep(2)

    container = docker_runtime._get_container(sklearn_model.details)
    assert container.status == "running"

    sklearn_model.undeploy()


@pytest.mark.parametrize(
    "x_input",
    [[[1, 2, 3, 4]], np.array([[1, 2, 3, 4]]), {"data": {"ndarray": [[1, 2, 3, 4]]}}],
)
def test_sklearn_docker(sklearn_model: Model, x_input):
    sklearn_model.deploy()
    time.sleep(2)

    y_pred = sklearn_model(x_input)

    np.testing.assert_allclose(y_pred, [[0, 0, 0.99]], atol=1e-2)

    sklearn_model.undeploy()


def test_undeploy_docker(sklearn_model: Model, docker_runtime: SeldonDockerRuntime):
    sklearn_model.deploy()
    time.sleep(2)

    sklearn_model.undeploy()

    with pytest.raises(docker.errors.NotFound):
        docker_runtime._get_container(sklearn_model.details)
