import os
import yaml
import time
import requests

from kubernetes import client, config
from kubernetes.client.rest import ApiException
from typing import Any

from tempo.seldon.endpoint import Endpoint
from tempo.seldon.protocol import SeldonProtocol
from tempo.seldon.specs import KubernetesSpec
from tempo.serve.runtime import Runtime
from tempo.utils import logger
from tempo.serve.metadata import ModelDetails, KubernetesOptions

ENV_K8S_SERVICE_HOST = "KUBERNETES_SERVICE_HOST"


class SeldonKubernetesRuntime(Runtime):
    def __init__(self, k8s_options: KubernetesOptions = None, protocol=None):
        if k8s_options is None:
            k8s_options = KubernetesOptions()
        self.k8s_options = k8s_options
        self.create_k8s_client()
        if protocol is None:
            self.protocol = SeldonProtocol()
        else:
            self.protocol = protocol

    def get_protocol(self):
        return self.protocol

    def create_k8s_client(self):
        inside_cluster = os.getenv(ENV_K8S_SERVICE_HOST)
        if inside_cluster:
            logger.debug("Loading cluster local config")
            config.load_incluster_config()
        else:
            logger.debug("Loading external kubernetes config")
            config.load_kube_config()

    def get_endpoint(self, model_details: ModelDetails) -> str:
        endpoint = Endpoint(
            model_details.name, self.k8s_options.namespace, self.protocol
        )
        return endpoint.get_url(model_details)

    def remote(self, model_details: ModelDetails, *args, **kwargs) -> Any:
        protocol = self.get_protocol()
        req = protocol.to_protocol_request(*args, **kwargs)
        endpoint = self.get_endpoint(model_details)
        response_raw = requests.post(endpoint, json=req)
        return protocol.from_protocol_response(
            response_raw.json(), model_details.outputs
        )

    def undeploy(self, model_details: ModelDetails):
        api_instance = client.CustomObjectsApi()
        api_instance.delete_namespaced_custom_object(
            "machinelearning.seldon.io",
            "v1",
            self.k8s_options.namespace,
            "seldondeployments",
            model_details.name,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )

    def deploy(self, model_details: ModelDetails):
        k8s_spec = KubernetesSpec(model_details, self.protocol, self.k8s_options)
        model_spec = k8s_spec.spec
        logger.debug(model_spec)

        api_instance = client.CustomObjectsApi()

        try:
            existing = api_instance.get_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                self.k8s_options.namespace,
                "seldondeployments",
                model_details.name,
            )
            model_spec["metadata"]["resourceVersion"] = existing["metadata"][
                "resourceVersion"
            ]
            api_instance.replace_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                self.k8s_options.namespace,
                "seldondeployments",
                model_details.name,
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

    def wait_ready(self, model_details: ModelDetails, timeout_secs=None) -> bool:
        ready = False
        t0 = time.time()
        while not ready:
            api_instance = client.CustomObjectsApi()
            existing = api_instance.get_namespaced_custom_object(
                "machinelearning.seldon.io",
                "v1",
                self.k8s_options.namespace,
                "seldondeployments",
                model_details.name,
            )
            if "status" in existing and "state" in existing["status"]:
                ready = existing["status"]["state"] == "Available"
            if timeout_secs is not None:
                t1 = time.time()
                if t1 - t0 > timeout_secs:
                    return ready
        return ready

    def to_k8s_yaml(self, model_details: ModelDetails) -> str:
        k8s_spec = KubernetesSpec(model_details, self.protocol, self.k8s_options)
        return yaml.safe_dump(k8s_spec.spec)
