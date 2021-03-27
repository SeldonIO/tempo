import os
import time
from typing import Any

import requests
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from tempo.seldon.endpoint import Endpoint
from tempo.seldon.specs import KubernetesSpec
from tempo.serve.metadata import KubernetesOptions
from tempo.serve.runtime import Runtime, ModelSpec
from tempo.utils import logger

ENV_K8S_SERVICE_HOST = "KUBERNETES_SERVICE_HOST"


class SeldonKubernetesRuntime(Runtime):
    def __init__(self, k8s_options: KubernetesOptions = None):
        if k8s_options is None:
            k8s_options = KubernetesOptions()
        self.k8s_options = k8s_options

    def create_k8s_client(self):
        inside_cluster = os.getenv(ENV_K8S_SERVICE_HOST)
        if inside_cluster:
            logger.debug("Loading cluster local config")
            config.load_incluster_config()
        else:
            logger.debug("Loading external kubernetes config")
            config.load_kube_config()

    def get_endpoint(self, model_spec: ModelSpec) -> str:
        self.create_k8s_client()
        endpoint = Endpoint(model_spec.model_details.name, self.k8s_options.namespace, model_spec.protocol)
        return endpoint.get_url(model_spec.model_details)

    def remote(self, model_spec: ModelSpec, *args, **kwargs) -> Any:
        req = model_spec.protocol.to_protocol_request(*args, **kwargs)
        endpoint = self.get_endpoint(model_spec)
        logger.debug("Endpoint is ", endpoint)
        response_raw = requests.post(endpoint, json=req)
        return model_spec.protocol.from_protocol_response(response_raw.json(), model_spec.model_details.outputs)

    def undeploy(self, model_spec: ModelSpec):
        self.create_k8s_client()
        api_instance = client.CustomObjectsApi()
        api_instance.delete_namespaced_custom_object(
            "machinelearning.seldon.io",
            "v1",
            self.k8s_options.namespace,
            "seldondeployments",
            model_spec.model_details.name,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )

    def deploy(self, model_details: ModelSpec):
        self.create_k8s_client()
        k8s_spec = KubernetesSpec(model_details, self.k8s_options)
        model_spec = k8s_spec.spec
        logger.debug(model_spec)

        api_instance = client.CustomObjectsApi()

        try:
            existing = api_instance.get_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                self.k8s_options.namespace,
                "seldondeployments",
                model_details.model_details.name,
            )
            model_spec["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
            api_instance.replace_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                self.k8s_options.namespace,
                "seldondeployments",
                model_details.model_details.name,
                model_spec,
            )
        except ApiException as e:
            if e.status == 404:
                api_instance.create_namespaced_custom_object(
                    "machinelearning.seldon.io",
                    "v1",
                    self.k8s_options.namespace,
                    "seldondeployments",
                    model_spec,
                )
            else:
                raise e

    def wait_ready(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        self.create_k8s_client()
        ready = False
        t0 = time.time()
        while not ready:
            api_instance = client.CustomObjectsApi()
            existing = api_instance.get_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                self.k8s_options.namespace,
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

    def to_k8s_yaml(self, model_spec: ModelSpec) -> str:
        k8s_spec = KubernetesSpec(model_spec, self.k8s_options)
        return yaml.safe_dump(k8s_spec.spec)
