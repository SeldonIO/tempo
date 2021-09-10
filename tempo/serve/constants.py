from .metadata import ModelFramework

SKLEARN_MODEL = "sklearn"
XGBOOST_MODEL = "xgboost"
TENSORFLOW_MODEL = "tensorflow"
PYTORCH_MODEL = "pytorch"
ONNX_MODEL = "onnx"

DefaultModelFilename = "model.pickle"
DefaultRemoteFilename = "remote.pickle"
DefaultEnvFilename = "environment.tar.gz"

MLServerEnvDeps = ["mlserver==0.4.1"]

# for each runtime we require separate dependencies to be packed
MLServerRuntimeEnvDeps = {
    ModelFramework.MLFlow: ["mlserver-mlflow==0.4.1"],
}

DefaultCondaFile = "conda.yaml"

ENV_K8S_SERVICE_HOST = "KUBERNETES_SERVICE_HOST"

ENV_TEMPO_RUNTIME_OPTIONS = "TEMPO_RUNTIME_OPTIONS"

DefaultInsightsServiceName = "insights-dumper"
DefaultInsightsPort = 8080
# TODO: Build our own tempo message dumper image
DefaultInsightsImage = "mendhak/http-https-echo:18"

DefaultInsightsLocalEndpoint = "http://0.0.0.0:8080"
DefaultInsightsDockerEndpoint = f"http://{DefaultInsightsServiceName}:8080"
DefaultSeldonSystemNamespace = "seldon-system"
DefaultInsightsK8sEndpoint = f"http://{DefaultInsightsServiceName}.{DefaultSeldonSystemNamespace}:8080"


DefaultRedisServiceName = "redis-master"
DefaultRedisImage = "docker.io/redis:6.0.5"

DefaultRedisPort = 6379
DefaultRedisLocalHost = "0.0.0.0"
DefaultRedisDockerHost = f"{DefaultRedisServiceName}"
DefaultRedisK8sHost = f"{DefaultRedisServiceName}.{DefaultSeldonSystemNamespace}"
