import time
import os
import pytest
import docker
import numpy as np

from tempo.serve.pipeline import Pipeline
from tempo.seldon.docker import SeldonDockerRuntime
from tempo.serve.utils import pipeline


def test_deploy_pipeline_docker(
    inference_pipeline: Pipeline,
    docker_runtime: SeldonDockerRuntime,
    docker_runtime_v2: SeldonDockerRuntime,
):
    for model in inference_pipeline._models:
        container = docker_runtime._get_container(model.details)
        assert container.status == "running"

    pipeline_container = docker_runtime_v2._get_container(inference_pipeline.details)
    assert pipeline_container.status == "running"


@pytest.mark.parametrize(
    "x_input",
    [
        pytest.param(
            [[1, 2, 3, 4]],
            marks=pytest.mark.skip(reason="not supported with KFServingV2Protocol"),
        ),
        np.array([[0.5, 2, 3, 4]]),
        pytest.param(
            {
                "inputs": [
                    {
                        "name": "payload",
                        "shape": [1, 4],
                        "datatype": "INT32",
                        "data": [[0.4, 2, 3, 4]],
                    }
                ]
            },
            marks=pytest.mark.skip(reason="not supported with KFServingV2Protocol"),
        ),
    ],
)
def test_pipeline_remote(inference_pipeline: Pipeline, x_input):
    y_pred = inference_pipeline.remote(payload=x_input)

    np.testing.assert_allclose(y_pred, [2.0], atol=1e-2)


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


async def test_pipeline_save_run(inference_pipeline: Pipeline, tmp_path: str):

    loaded_pipeline = inference_pipeline.load()

    y_pred = loaded_pipeline(np.array([[4.9, 3.1, 1.5, 0.2]]))

    np.testing.assert_allclose(y_pred, [[0.8, 0.19, 0.01]], atol=1e-2)


def test_pipeline_save(docker_runtime_v2: SeldonDockerRuntime):
    @pipeline(
        name="classifier",
        runtime=docker_runtime_v2,
        local_folder="/tmp/tempo-pipeline"
    )
    def mypipeline(payload: np.ndarray) -> np.ndarray:
        return payload

    mypipeline.save()