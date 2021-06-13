import time

import numpy as np
import pytest

from tempo.seldon.docker import SeldonDockerRuntime


def test_deploy_docker(sklearn_model_deployed, runtime: SeldonDockerRuntime):

    time.sleep(2)

    container = runtime._get_container(sklearn_model_deployed.model_spec)
    assert container.status == "running"


@pytest.mark.parametrize(
    "x_input",
    [[[1, 2, 3, 4]], np.array([[1, 2, 3, 4]]), {"data": {"ndarray": [[1, 2, 3, 4]]}}],
)
def test_sklearn_docker(sklearn_model_deployed, x_input):
    time.sleep(2)

    y_pred = sklearn_model_deployed.predict(x_input)

    np.testing.assert_allclose(y_pred, [[0, 0, 0.99]], atol=1e-2)
