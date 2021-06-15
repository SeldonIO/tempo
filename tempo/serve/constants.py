SKLEARN_MODEL = "sklearn"
XGBOOST_MODEL = "xgboost"
TENSORFLOW_MODEL = "tensorflow"
PYTORCH_MODEL = "pytorch"
ONNX_MODEL = "onnx"

DefaultModelFilename = "model.pickle"
DefaultRemoteFilename = "remote.pickle"
DefaultEnvFilename = "environment.tar.gz"

MLServerEnvDeps = ["mlserver==0.3.2"]
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
