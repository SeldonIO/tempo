import functools
from typing import Any
from metaflow import S3, FlowSpec, IncludeFile, Parameter, conda, step

pipeline_folder_name = "classifier"
sklearn_folder_name = "sklearn"
xgboost_folder_name = "xgboost"


def script_path(filename):
    """
    A convenience function to get the absolute path to a file in this
    tutorial's directory. This allows the tutorial to be launched from any
    directory.

    """
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


def pip(libraries, test_index=False):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            import subprocess
            import sys

            for library, version in libraries.items():
                if not test_index:
                    print("Pip Install:", library, version)
                    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                                    library + "==" + version])
                else:
                    print("Pip Test Install:", library, version)
                    subprocess.run([sys.executable, "-m", "pip", "install", "--quiet",
                                    "--index-url", "https://test.pypi.org/simple/",
                                    "--extra-index-url", "https://pypi.org/simple",
                                    library + "==" + version])
            return function(*args, **kwargs)

        return wrapper

    return decorator


def save_bytes_local(model: Any, model_name: str):
    import os
    import tempfile

    folder = tempfile.mkdtemp()
    print(folder)
    local_model_path = os.path.join(folder, model_name)
    with open(local_model_path, "wb") as f:
        f.write(model)
    return folder


def gke_authenticate(kubeconfig: IncludeFile, gsa_key: IncludeFile):
    import os
    import tempfile
    from importlib import reload
    import kubernetes.config.kube_config

    k8s_folder = tempfile.mkdtemp()
    kubeconfig_path = os.path.join(k8s_folder, "kubeconfig.yaml")
    with open(kubeconfig_path, "w") as f:
        f.write(kubeconfig)
    gsa_key_path = os.path.join(k8s_folder, "gsa-key.json")
    with open(gsa_key_path, "w") as f:
        f.write(gsa_key)
    os.environ["KUBECONFIG"] = kubeconfig_path
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gsa_key_path
    reload(kubernetes.config.kube_config)  # need to refresh environ variables used for kubeconfig


def create_s3_folder(flow_spec: FlowSpec, folder: str) -> str:
    from metaflow import S3
    import os
    with S3(run=flow_spec) as s3:
        return os.path.split(s3.put(folder + "/.keep", "keep"))[0]


def upload_s3_folder(flow_spec: FlowSpec, s3_folder: str, path: str):
    import os

    artifact_files = [
        (os.path.join(s3_folder, f), os.path.join(path, f))
        for f in os.listdir(path)
    ]
    with S3(run=flow_spec) as s3:
        s3.put_files(iter(artifact_files))


def save_pipeline(pipeline, folder: str, conda_env: IncludeFile):
    from tempo import save
    import os

    conda_env_path = os.path.join(folder, "conda.yaml")
    with open(conda_env_path, "w") as f:
        f.write(conda_env)
    save(pipeline)

class IrisFlow(FlowSpec):
    """


    The flow performs the following steps:

    1) Load Iris Data
    2) Train SKLearn LR Model
    3) Train XGBoost LR Model
    4) Create and deploy Tempo artifacts
    """

    conda_env = IncludeFile(
        "conda_env", help="The path to conda environment for classifier", default=script_path("conda.yaml")
    )
    kubeconfig = IncludeFile("kubeconfig", help="The path to kubeconfig", default=script_path("kubeconfig.yaml"))
    gsa_key = IncludeFile(
        "gsa_key", help="The path to google service account json", default=script_path("gsa-key.json")
    )
    tempo_on_docker = Parameter("tempo-on-docker", help="Whether to deploy Tempo artifacts to Docker", default=False)
    k8s_provider = Parameter("k8s_provider", help="kubernetes provider. Needed for non local run to deploy", default="gke")

    @conda(libraries={"scikit-learn": "0.24.1"})
    @step
    def start(self):
        # pylint: disable=no-member
        from sklearn import datasets

        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target
        self.next(self.train_sklearn, self.train_xgboost)

    @conda(libraries={"scikit-learn": "0.24.1"})
    @step
    def train_sklearn(self):
        from joblib import dump
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e5)
        lr.fit(self.X, self.y)
        dump(lr, script_path("model.joblib"))
        with open(script_path("model.joblib"), "rb") as fh:
            self.buffered_lr_model = fh.read()

        self.next(self.join)

    @conda(libraries={"xgboost": "1.4.0"})
    @step
    def train_xgboost(self):
        from xgboost import XGBClassifier

        xgb = XGBClassifier()
        xgb.fit(self.X, self.y)
        xgb.save_model(script_path("model.bst"))
        with open(script_path("model.bst"), "rb") as fh:
            self.buffered_xgb_model = fh.read()
        self.next(self.join)

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs)

        self.next(self.tempo)

    def create_tempo_artifacts(self):
        import tempfile
        from deploy import get_tempo_artifacts
        # Store models to local artifact locations
        local_sklearn_path = save_bytes_local(self.buffered_lr_model, "model.joblib")
        local_xgb_path = save_bytes_local(self.buffered_xgb_model, "model.bst")
        local_pipeline_path = tempfile.mkdtemp()
        # Create S3 folders for artifacts
        classifier_url = create_s3_folder(self, pipeline_folder_name)
        sklearn_url = create_s3_folder(self, sklearn_folder_name)
        xgboost_url = create_s3_folder(self, xgboost_folder_name)
        classifier, sklearn_model, xgboost_model = get_tempo_artifacts(
            local_sklearn_path,
            local_xgb_path,
            local_pipeline_path,
            sklearn_url,
            xgboost_url,
            classifier_url
        )
        # Create pipeline artifacts
        save_pipeline(classifier, local_pipeline_path, self.conda_env)
        # Upload artifacts to S3
        upload_s3_folder(self, pipeline_folder_name, local_pipeline_path)
        upload_s3_folder(self, sklearn_folder_name, local_sklearn_path)
        upload_s3_folder(self, xgboost_folder_name, local_xgb_path)
        return classifier

    def deploy_tempo_local(self, classifier):
        from tempo import deploy_local
        from tempo.serve.deploy import get_client
        import time
        import numpy as np

        remote_model = deploy_local(classifier)
        self.client_model = get_client(remote_model)
        time.sleep(10)
        print(self.client_model.predict(np.array([[1, 2, 3, 4]])))

    def deploy_tempo_remote(self, classifier):
        from tempo import deploy_remote
        from tempo.serve.metadata import SeldonCoreOptions
        from tempo.serve.deploy import get_client
        import time
        import numpy as np

        if self.k8s_provider == "gke":
            gke_authenticate(self.kubeconfig, self.gsa_key)
        else:
            raise Exception(f"Unknown Kubernetes Provider {self.k8s_provider}")

        runtime_options = SeldonCoreOptions(**{
            "remote_options": {
                "namespace": "production",
                "authSecretName": "s3-secret"
            }
        })

        remote_model = deploy_remote(classifier, options=runtime_options)
        self.client_model = get_client(remote_model)
        time.sleep(10)
        print(self.client_model.predict(np.array([[1, 2, 3, 4]])))

    @conda(libraries={"numpy": "1.19.5"})
    @pip(libraries={"mlops-tempo": "0.4.0.dev3", "conda_env": "2.4.2"}, test_index=True)
    @step
    def tempo(self):
        classifier = self.create_tempo_artifacts()

        if self.tempo_on_docker:
            self.deploy_tempo_local(classifier)
        else:
            self.deploy_tempo_remote(classifier)

        self.next(self.end)



    @step
    def end(self):
        pass


if __name__ == "__main__":
    IrisFlow()
