import time
import os
import pytest
import docker
import numpy as np

from tempo.serve.pipeline import Pipeline
from tempo.seldon.docker import SeldonDockerRuntime


def test_deploy_pipeline_docker(
    inference_pipeline: Pipeline, docker_runtime: SeldonDockerRuntime
):
    for model in inference_pipeline._models:
        container = docker_runtime._get_container(model.details)
        assert container.status == "running"


@pytest.mark.parametrize(
    "x_input",
    [
        [[1, 2, 3, 4]],
        np.array([[0.5, 2, 3, 4]]),
        {"data": {"ndarray": [[0.4, 2, 3, 4]]}},
    ],
)
def test_pipeline_docker(inference_pipeline: Pipeline, x_input):
    y_pred = inference_pipeline(x_input)

    np.testing.assert_allclose(y_pred, [2.0], atol=1e-2)


@pytest.mark.parametrize(
    "x_input, expected",
    [({"data": {"ndarray": [[0.4, 2, 3, 4]]}}, {"data": {"ndarray": [2.0]}})],
)
def test_seldon_pipeline_request_docker(
    inference_pipeline: Pipeline, x_input, expected
):
    y_pred = inference_pipeline.request(x_input)

    assert y_pred == expected


@pytest.mark.parametrize(
    "x_input, expected",
    [
        (
            {
                "inputs": [
                    {
                        "name": "input0",
                        "datatype": "FP64",
                        "shape": [1, 4],
                        "data": [0.4, 2, 3, 4],
                    }
                ]
            },
            {
                "model_name": "inference-pipeline",
                "outputs": [
                    {"name": "output0", "datatype": "FP64", "shape": [1], "data": [2.0]}
                ],
            },
        )
    ],
)
def test_v2_pipeline_request_docker(inference_pipeline_v2: Pipeline, x_input, expected):
    y_pred = inference_pipeline_v2.request(x_input)

    assert y_pred == expected


def test_undeploy_pipeline_docker(
    inference_pipeline: Pipeline, docker_runtime: SeldonDockerRuntime
):
    inference_pipeline.undeploy()

    for model in inference_pipeline._models:
        with pytest.raises(docker.errors.NotFound):
            docker_runtime._get_container(model.details)


async def test_pipeline_save(inference_pipeline: Pipeline, tmp_path: str):
    pipeline_path = os.path.join(tmp_path, "pipeline.pickle")
    inference_pipeline.save(pipeline_path)

    loaded_pipeline = Pipeline.load(pipeline_path)

    y_pred = loaded_pipeline(np.array([[4.9, 3.1, 1.5, 0.2]]))

    np.testing.assert_allclose(y_pred, [[0.8, 0.19, 0.01]], atol=1e-2)
