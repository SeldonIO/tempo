import os
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException
import time
from typing import Dict, Any
import requests

from tempo.kfserving.endpoint import Endpoint
from tempo.kfserving.protocol import KFServingV1Protocol
from tempo.serve.runtime import Runtime
from tempo.utils import logger
from tempo.serve.metadata import ModelDetails, ModelFramework, KubernetesOptions

ENV_K8S_SERVICE_HOST = "KUBERNETES_SERVICE_HOST"

Implementations = {
    ModelFramework.SKLearn: "sklearn",
    ModelFramework.XGBoost: "xgboost",
    ModelFramework.Tensorflow: "tensorflow",
    ModelFramework.PyTorch: "triton",
    ModelFramework.ONNX: "triton",
    ModelFramework.TensorRT: "triton",
}


class KFServingKubernetesRuntime(Runtime):
    def __init__(self, k8s_options: KubernetesOptions = None, protocol=None):
        if k8s_options is None:
            k8s_options = KubernetesOptions()
        self.k8s_options = k8s_options
        self.create_k8s_client()
        if protocol is None:
            self.protocol = KFServingV1Protocol()
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
            model_details, self.k8s_options.namespace, self.protocol
        )
        return endpoint.get_url()

    def get_headers(self, model_details: ModelDetails) -> Dict[str, str]:
        endpoint = Endpoint(
            model_details, self.k8s_options.namespace, self.protocol
        )
        service_host = endpoint.get_service_host()
        return {"Host":service_host}

    def remote(self, model_details: ModelDetails, *args, **kwargs) -> Any:
        protocol = self.get_protocol()
        req = protocol.to_protocol_request(*args, **kwargs)
        endpoint = self.get_endpoint(model_details)
        headers = self.get_headers(model_details)
        response_raw = requests.post(endpoint, json=req, headers=headers)
        return protocol.from_protocol_response(response_raw.json(), model_details.outputs)

    def undeploy(self, model_details: ModelDetails):
        api_instance = client.CustomObjectsApi()
        api_instance.delete_namespaced_custom_object(
            "serving.kubeflow.org",
            "v1alpha2",
            self.k8s_options.namespace,
            "inferenceservices",
            model_details.name,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )

    def deploy(self, model_details: ModelDetails):
        model_spec = self._get_spec(model_details)
        logger.debug(model_spec)

        api_instance = client.CustomObjectsApi()

        try:
            existing = api_instance.get_namespaced_custom_object(
                "serving.kubeflow.org",
                "v1alpha2",
                self.k8s_options.namespace,
                "inferenceservices",
                model_details.name,
            )
            model_spec["metadata"]["resourceVersion"] = existing["metadata"][
                "resourceVersion"
            ]
            api_instance.replace_namespaced_custom_object(
                "serving.kubeflow.org",
                "v1alpha2",
                self.k8s_options.namespace,
                "inferenceservices",
                model_details.name,
                model_spec,
            )
        except ApiException as e:
            if e.status == 404:
                api_instance.create_namespaced_custom_object(
                    "serving.kubeflow.org",
                    "v1alpha2",
                    self.k8s_options.namespace,
                    "inferenceservices",
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
                "serving.kubeflow.org",
                "v1alpha2",
                self.k8s_options.namespace,
                "inferenceservices",
                model_details.name,
            )
            default_ready = False
            routes_ready = False
            for item in existing["status"]["conditions"]:
                if item["type"] == "DefaultPredictorReady":
                    default_ready = item["status"] == "True"
                elif item["type"] == "RoutesReady":
                    routes_ready = item["status"] == "True"
            ready = default_ready and routes_ready
            if timeout_secs is not None:
                t1 = time.time()
                if t1 - t0 > timeout_secs:
                    return ready
        return ready

    def _get_spec(self, model_details: ModelDetails) -> dict:
        model_implementation = Implementations[model_details.platform]
        return {
            "apiVersion": "serving.kubeflow.org/v1alpha2",
            "kind": "InferenceService",
            "metadata": {
                "name": model_details.name,
                "namespace": self.k8s_options.namespace,
            },
            "spec": {
                "default": {
                  "predictor":
                  {
                    model_implementation: {
                      "storageUri": model_details.uri
                     },
                  },
                },
            },
        }

    def to_k8s_yaml(self, model_details: ModelDetails) -> str:
        d = self._get_spec(model_details)
        return yaml.safe_dump(d)
