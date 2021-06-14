import json
import time
from typing import Optional, Sequence

import yaml
from kubernetes import client
from kubernetes.client.rest import ApiException

from tempo.k8s.constants import TempoK8sLabel, TempoK8sModelSpecAnnotation
from tempo.k8s.utils import create_k8s_client
from tempo.seldon.endpoint import Endpoint
from tempo.seldon.specs import KubernetesSpec
from tempo.serve.base import DeployedModel, ModelSpec, Runtime
from tempo.serve.metadata import RuntimeOptions
from tempo.serve.stub import deserialize
from tempo.utils import logger


class SeldonCoreOptions(RuntimeOptions):
    runtime: str = "tempo.seldon.SeldonKubernetesRuntime"


class SeldonKubernetesRuntime(Runtime):
    def __init__(self, runtime_options: Optional[RuntimeOptions] = None):
        if runtime_options is None:
            runtime_options = RuntimeOptions()
        runtime_options.runtime = "tempo.seldon.SeldonKubernetesRuntime"
        super().__init__(runtime_options)

    def get_endpoint_spec(self, model_spec: ModelSpec) -> str:
        create_k8s_client()
        endpoint = Endpoint()
        return endpoint.get_url(model_spec)

    def undeploy_spec(self, model_spec: ModelSpec):
        create_k8s_client()
        api_instance = client.CustomObjectsApi()
        api_instance.delete_namespaced_custom_object(
            "machinelearning.seldon.io",
            "v1",
            model_spec.runtime_options.k8s_options.namespace,
            "seldondeployments",
            model_spec.model_details.name,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )

    def deploy_spec(self, model_spec: ModelSpec):
        create_k8s_client()
        k8s_specer = KubernetesSpec(model_spec)
        k8s_spec = k8s_specer.spec
        logger.debug(k8s_spec)

        api_instance = client.CustomObjectsApi()

        try:
            existing = api_instance.get_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                model_spec.runtime_options.k8s_options.namespace,
                "seldondeployments",
                model_spec.model_details.name,
            )
            k8s_spec["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
            api_instance.replace_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                model_spec.runtime_options.k8s_options.namespace,
                "seldondeployments",
                model_spec.model_details.name,
                k8s_spec,
            )
        except ApiException as e:
            if e.status == 404:
                api_instance.create_namespaced_custom_object(
                    "machinelearning.seldon.io",
                    "v1",
                    model_spec.runtime_options.k8s_options.namespace,
                    "seldondeployments",
                    k8s_spec,
                )
            else:
                raise e

    def wait_ready_spec(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        create_k8s_client()
        ready = False
        t0 = time.time()
        while not ready:
            api_instance = client.CustomObjectsApi()
            existing = api_instance.get_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                model_spec.runtime_options.k8s_options.namespace,
                "seldondeployments",
                model_spec.model_details.name,
            )
            if "status" in existing and "state" in existing["status"]:
                ready = existing["status"]["state"] == "Available"
            if timeout_secs is not None:
                t1 = time.time()
                if t1 - t0 > timeout_secs:
                    return ready
        return ready

    def to_k8s_yaml_spec(self, model_spec: ModelSpec) -> str:
        k8s_spec = KubernetesSpec(model_spec)
        return yaml.safe_dump(k8s_spec.spec)

    def list_models(self, namespace: Optional[str] = None) -> Sequence[DeployedModel]:
        create_k8s_client()
        api_instance = client.CustomObjectsApi()

        if namespace is None and self.runtime_options is not None:
            namespace = self.runtime_options.k8s_options.namespace

        if namespace is None:
            return []

        try:
            models = []
            response = api_instance.list_namespaced_custom_object(
                group="machinelearning.seldon.io",
                version="v1",
                namespace=namespace,
                plural="seldondeployments",
                label_selector=TempoK8sLabel + "=true",
            )
            for model in response["items"]:
                metadata = model["metadata"]["annotations"][TempoK8sModelSpecAnnotation]
                remote_model = deserialize(json.loads(metadata))
                models.append(remote_model)
            return models
        except ApiException as e:
            if e.status == 404:
                return []
            else:
                raise e
