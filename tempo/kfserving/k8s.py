import os
import time
from typing import Any, Dict

import requests
import yaml
from kubernetes import client, config
from kubernetes.client.rest import ApiException

from tempo.kfserving.endpoint import Endpoint
from tempo.kfserving.protocol import KFServingV2Protocol
from tempo.seldon.specs import DefaultModelsPath, DefaultServiceAccountName
from tempo.serve.metadata import KubernetesOptions, ModelFramework
from tempo.serve.runtime import Runtime, ModelSpec
from tempo.utils import logger
from tempo.seldon.constants import MLSERVER_IMAGE
from tempo.serve.remote import Remote

DefaultHTTPPort = "8080"
DefaultGRPCPort = "9000"

ENV_K8S_SERVICE_HOST = "KUBERNETES_SERVICE_HOST"

Implementations = {
    ModelFramework.SKLearn: "sklearn",
    ModelFramework.XGBoost: "xgboost",
    ModelFramework.Tensorflow: "tensorflow",
    ModelFramework.PyTorch: "triton",
    ModelFramework.ONNX: "triton",
    ModelFramework.TensorRT: "triton",
}


class KFServingKubernetesRuntime(Runtime, Remote):
    def __init__(self, k8s_options: KubernetesOptions = None, protocol=None):
        if k8s_options is None:
            k8s_options = KubernetesOptions()
        self.k8s_options = k8s_options
        if self.k8s_options.serviceAccountName is None:
            self.serviceAccountName = DefaultServiceAccountName
        else:
            self.serviceAccountName = self.k8s_options.serviceAccountName
        self.inside_cluster = self.create_k8s_client()

    def create_k8s_client(self):
        inside_cluster = os.getenv(ENV_K8S_SERVICE_HOST)
        if inside_cluster:
            print("Loading cluster local config")
            config.load_incluster_config()
            return True
        else:
            print("Loading external kubernetes config")
            config.load_kube_config()
            return False

    def get_endpoint_spec(self, model_spec: ModelSpec) -> str:
        endpoint = Endpoint(model_spec, self.k8s_options.namespace)
        return endpoint.get_url()

    def get_headers(self, model_spec: ModelSpec) -> Dict[str, str]:
        self.inside_cluster = self.create_k8s_client()
        if not self.inside_cluster:
            endpoint = Endpoint(model_spec, self.k8s_options.namespace)
            service_host = endpoint.get_service_host()
            return {"Host": service_host}
        else:
            return {}

    def remote(self, model_spec: ModelSpec, *args, **kwargs) -> Any:
        req = model_spec.protocol.to_protocol_request(*args, **kwargs)
        endpoint = self.get_endpoint_spec(model_spec)
        print("Endpoint is ", endpoint)
        headers = self.get_headers(model_spec, )
        print("Headers are", headers)
        response_raw = requests.post(endpoint, json=req, headers=headers)
        if response_raw.status_code == 200:
            return model_spec.protocol.from_protocol_response(response_raw.json(), model_spec.model_details.outputs)
        else:
            raise ValueError("Bad return code", response_raw.status_code, response_raw.text)

    def undeploy_spec(self, model_spec: ModelSpec):
        api_instance = client.CustomObjectsApi()
        api_instance.delete_namespaced_custom_object(
            "serving.kubeflow.org",
            "v1beta1",
            self.k8s_options.namespace,
            "inferenceservices",
            model_spec.model_details.name,
            body=client.V1DeleteOptions(propagation_policy="Foreground"),
        )

    def deploy_spec(self, model_spec: ModelSpec):
        spec = self._get_spec(model_spec)
        logger.debug(model_spec)

        api_instance = client.CustomObjectsApi()

        try:
            existing = api_instance.get_namespaced_custom_object(
                "serving.kubeflow.org",
                "v1beta1",
                self.k8s_options.namespace,
                "inferenceservices",
                model_spec.model_details.name,
            )
            model_spec["metadata"]["resourceVersion"] = existing["metadata"]["resourceVersion"]
            api_instance.replace_namespaced_custom_object(
                "serving.kubeflow.org",
                "v1beta1",
                self.k8s_options.namespace,
                "inferenceservices",
                model_spec.model_details.name,
                spec,
            )
        except ApiException as e:
            if e.status == 404:
                api_instance.create_namespaced_custom_object(
                    "serving.kubeflow.org",
                    "v1beta1",
                    self.k8s_options.namespace,
                    "inferenceservices",
                    spec,
                )
            else:
                raise e

    def wait_ready_spec(self, model_spec: ModelSpec, timeout_secs=None) -> bool:
        ready = False
        t0 = time.time()
        while not ready:
            api_instance = client.CustomObjectsApi()
            existing = api_instance.get_namespaced_custom_object(
                "serving.kubeflow.org",
                "v1beta1",
                self.k8s_options.namespace,
                "inferenceservices",
                model_spec.model_details.name,
            )
            default_ready = False
            routes_ready = False
            conf_ready = False
            ingress_ready = False
            if "status" in existing and "conditions" in existing["status"]:
                for item in existing["status"]["conditions"]:
                    if item["type"] == "PredictorReady":
                        default_ready = item["status"] == "True"
                    elif item["type"] == "PredictorRouteReady":
                        routes_ready = item["status"] == "True"
                    elif item["type"] == "IngressReady":
                        ingress_ready = item["status"] == "True"
                    elif item["type"] == "PredictorConfigurationReady":
                        conf_ready = item["status"] == "True"
                ready = default_ready and routes_ready and ingress_ready and conf_ready
            if timeout_secs is not None:
                t1 = time.time()
                if t1 - t0 > timeout_secs:
                    return ready
            time.sleep(1)
        return ready

    def _get_spec(self, model_spec: ModelSpec) -> dict:
        if model_spec.model_details.platform == ModelFramework.TempoPipeline:
            return {
                "apiVersion": "serving.kubeflow.org/v1beta1",
                "kind": "InferenceService",
                "metadata": {
                    "name": model_spec.model_details.name,
                    "namespace": self.k8s_options.namespace,
                },
                "spec": {
                    "predictor": {
                        "serviceAccountName": self.serviceAccountName,
                        "containers": [
                            {
                                "image": MLSERVER_IMAGE,
                                "name": "mlserver",
                                "env": [
                                    {
                                        "name": "STORAGE_URI",
                                        "value": model_spec.model_details.uri,
                                    },
                                    {
                                        "name": "MLSERVER_HTTP_PORT",
                                        "value": DefaultHTTPPort,
                                    },
                                    {
                                        "name": "MLSERVER_GRPC_PORT",
                                        "value": DefaultGRPCPort,
                                    },
                                    {
                                        "name": "MLSERVER_MODEL_IMPLEMENTATION",
                                        "value": "mlserver_tempo.TempoModel",
                                    },
                                    {
                                        "name": "MLSERVER_MODEL_NAME",
                                        "value": model_spec.model_details.name,
                                    },
                                    {
                                        "name": "MLSERVER_MODEL_URI",
                                        "value": DefaultModelsPath,
                                    },
                                ],
                            },
                        ],
                    },
                },
            }
        elif model_spec.model_details.platform in Implementations:
            model_implementation = Implementations[model_spec.model_details.platform]
            spec = {
                "apiVersion": "serving.kubeflow.org/v1beta1",
                "kind": "InferenceService",
                "metadata": {
                    "name": model_spec.model_details.name,
                    "namespace": self.k8s_options.namespace,
                },
                "spec": {
                    "predictor": {
                        model_implementation: {"storageUri": model_spec.model_details.uri},
                    },
                },
            }
            if self.k8s_options.serviceAccountName is not None:
                spec["spec"]["predictor"]["serviceAccountName"] = self.k8s_options.serviceAccountName
            if isinstance(model_spec.protocol, KFServingV2Protocol):
                spec["spec"]["predictor"][model_implementation]["protocolVersion"] = "v2"
            return spec
        else:
            raise ValueError("Can't create spec for implementation ", model_spec.model_details.platform)

    def to_k8s_yaml_spec(self, model_spec: ModelSpec) -> str:
        d = self._get_spec(model_spec)
        return yaml.safe_dump(d)
