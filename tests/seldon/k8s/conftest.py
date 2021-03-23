import os
import time
import uuid
from typing import Generator

import pytest
from kubernetes import client, config
from kubernetes.utils.create_from_yaml import create_from_yaml

from tempo import Model, Pipeline
from tempo.kfserving import KFServingV2Protocol
from tempo.seldon import SeldonKubernetesRuntime
from tempo.serve.metadata import KubernetesOptions

from ...conftest import TESTDATA_PATH

K8S_NAMESPACE_PREFIX = "test-tempo-"


@pytest.fixture
def namespace() -> Generator[str, None, None]:
    unique_id = str(uuid.uuid4().fields[-1])[:5]
    ns_name = f"{K8S_NAMESPACE_PREFIX}{unique_id}"

    config.load_kube_config()
    core_api = client.CoreV1Api()

    namespace = client.V1Namespace(metadata=client.V1ObjectMeta(name=ns_name))
    core_api.create_namespace(namespace)

    # Create ServiceAccount and RBAC
    rbac_path = os.path.join(TESTDATA_PATH, "tempo-pipeline-rbac.yaml")
    api_client = client.ApiClient()
    create_from_yaml(api_client, rbac_path, namespace=ns_name)

    yield ns_name

    core_api.delete_namespace(ns_name)


@pytest.fixture
def runtime(namespace: str) -> SeldonKubernetesRuntime:
    return SeldonKubernetesRuntime(k8s_options=KubernetesOptions(namespace=namespace))


@pytest.fixture
def runtime_v2(namespace: str) -> SeldonKubernetesRuntime:
    return SeldonKubernetesRuntime(
        k8s_options=KubernetesOptions(namespace=namespace),
        protocol=KFServingV2Protocol(),
    )


@pytest.fixture
def sklearn_model(sklearn_model: Model, runtime: SeldonKubernetesRuntime) -> Generator[Model, None, None]:
    sklearn_model.set_runtime(runtime)

    sklearn_model.deploy()
    sklearn_model.wait_ready(timeout_secs=60)

    yield sklearn_model

    sklearn_model.undeploy()


@pytest.fixture
def xgboost_model(xgboost_model: Model, runtime: SeldonKubernetesRuntime) -> Generator[Model, None, None]:
    xgboost_model.set_runtime(runtime)
    xgboost_model.deploy()
    xgboost_model.wait_ready(timeout_secs=60)

    yield xgboost_model

    xgboost_model.undeploy()


@pytest.fixture
def inference_pipeline(
    inference_pipeline: Pipeline,
    runtime: SeldonKubernetesRuntime,
    runtime_v2: SeldonKubernetesRuntime,
) -> Generator[Pipeline, None, None]:
    inference_pipeline.set_runtime(runtime_v2)

    for model in inference_pipeline._models:
        model.set_runtime(runtime)

    inference_pipeline.save(save_env=False)
    inference_pipeline.upload()
    inference_pipeline.deploy()
    # TODO: Fix wait_ready for pipelines
    time.sleep(60)
    inference_pipeline.wait_ready(timeout_secs=60)

    yield inference_pipeline

    inference_pipeline.undeploy()
