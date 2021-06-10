SKLEARN_MODEL = "sklearn"
XGBOOST_MODEL = "xgboost"
TENSORFLOW_MODEL = "tensorflow"
PYTORCH_MODEL = "pytorch"
ONNX_MODEL = "onnx"

DefaultModelFilename = "model.pickle"
DefaultRemoteFilename = "remote.pickle"
DefaultEnvFilename = "environment.tar.gz"

# TODO: Update once tempo is published
MLServerEnvDeps = ["mlserver==0.3.1.dev7"]
DefaultCondaFile = "conda.yaml"

ENV_K8S_SERVICE_HOST = "KUBERNETES_SERVICE_HOST"

ENV_TEMPO_RUNTIME_OPTIONS = "TEMPO_RUNTIME_OPTIONS"

DefaultInsightsServiceName = "insights-dumper"
DefaultInsightsPort = 8080
DefaultInsightsLocalEndpoint = "http://0.0.0.0:8080"
DefaultInsightsDockerEndpoint = f"http://{DefaultInsightsServiceName}:8080"
DefaultInsightsImage = "mendhak/http-https-echo:18"

