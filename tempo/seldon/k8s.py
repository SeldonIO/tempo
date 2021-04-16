import os
import time
from typing import Any, Optional

import requests
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from tempo.seldon.endpoint import Endpoint
from tempo.seldon.specs import KubernetesSpec
from tempo.serve.constants import ENV_K8S_SERVICE_HOST
from tempo.serve.metadata import RuntimeOptions
from tempo.serve.remote import Remote
from tempo.serve.runtime import ModelSpec, Runtime
from tempo.utils import logger


class SeldonKubernetesRuntime(Runtime, Remote):
    def __init__(self, runtime_options: Optional[RuntimeOptions] = None):
        if runtime_options:
            runtime_options.runtime = "tempo.seldon.SeldonKubernetesRuntime"
        super().__init__(runtime_options)

    def create_k8s_client(self):
        inside_cluster = os.getenv(ENV_K8S_SERVICE_HOST)
        if inside_cluster:
            logger.debug("Loading cluster local config")
            config.load_incluster_config()
        else:
            logger.debug("Loading external kubernetes config")
            config.load_kube_config()

    def get_endpoint_spec(self, model_spec: ModelSpec) -> str:
        self.create_k8s_client()
        endpoint = Endpoint()
        return endpoint.get_url(model_spec)

    def remote(self, model_spec: ModelSpec, *args, **kwargs) -> Any:
        req = model_spec.protocol.to_protocol_request(*args, **kwargs)
        endpoint = self.get_endpoint_spec(model_spec)
        logger.debug("Endpoint is ", endpoint)
        response_raw = requests.post(endpoint, json=req, verify=model_spec.runtime_options.ingress_options.verify_ssl)
        return model_spec.protocol.from_protocol_response(response_raw.json(), model_spec.model_details.outputs)

    def undeploy_spec(self, model_spec: ModelSpec):
        self.create_k8s_client()
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
        self.create_k8s_client()
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
        self.create_k8s_client()
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
