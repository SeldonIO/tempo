import functools
from typing import Any

from metaflow import S3, FlowSpec, IncludeFile, Parameter, conda, step


def script_path(filename):
    """
    A convenience function to get the absolute path to a file in this
    tutorial's directory. This allows the tutorial to be launched from any
    directory.

    """
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


def pip(libraries):
    def decorator(function):
        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            import subprocess
            import sys

            for library, version in libraries.items():
                print("Pip Install:", library, version)
                subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", library + "==" + version])
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
        # shutil.copyfileobj(model, f)
    return folder


class IrisFlow(FlowSpec):
    """


    The flow performs the following steps:

    1) Load Iris Data
    """

    conda_env = IncludeFile(
        "conda_env", help="The path to conda environment for classifier", default=script_path("conda.yaml")
    )

    kubeconfig = IncludeFile("kubeconfig", help="The path to kubeconfig", default=script_path("kubeconfig.yaml"))

    gsa_key = IncludeFile(
        "gsa_key", help="The path to google service account json", default=script_path("gsa-key.json")
    )

    run_local = Parameter("local", help="Whether to run locally", default=False)

    k8s_provider = Parameter("k8s_provider", help="kubernetes provider", default="gke")

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

    @conda(libraries={"numpy": "1.19.5"})
    @pip(libraries={"mlops-tempo": "0.2.0", "conda_env": "2.4.2"})
    @step
    def tempo(self):
        import os
        import tempfile
        import time

        import numpy as np
        from deploy import get_tempo_artifacts

        from tempo import deploy
        from tempo.serve.loader import save

        # create the S3 folders for our 2 models and tempo pipeline and
        # store the model artifacts
        with S3(run=self) as s3:
            classifier_loc = s3.put("classifier/.keep", "keep")
            classifier_loc = os.path.split(classifier_loc)[0]
            sklearn_url = s3.put("sklearn/model.joblib", self.buffered_lr_model)
            xgboost_url = s3.put("xgboost/model.bst", self.buffered_xgb_model)
        # Store models to local artifact locations
        local_sklearn_path = save_bytes_local(self.buffered_lr_model, "model.joblib")
        local_xgb_path = save_bytes_local(self.buffered_xgb_model, "model.bst")
        classifier, sklearn_model, xgboost_model = get_tempo_artifacts(
            local_sklearn_path, sklearn_url, local_xgb_path, xgboost_url, classifier_loc
        )

        # Create k8s auth setup
        if not self.run_local:
            if self.k8s_provider == "gke":
                k8s_folder = tempfile.mkdtemp()
                kubeconfig_path = os.path.join(k8s_folder, "kubeconfig.yaml")
                with open(kubeconfig_path, "w") as f:
                    f.write(self.kubeconfig)
                gsa_key_path = os.path.join(k8s_folder, "gsa-key.json")
                with open(gsa_key_path, "w") as f:
                    f.write(self.gsa_key)
                os.environ["KUBECONFIG"] = kubeconfig_path
                os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = gsa_key_path
                from importlib import reload

                import kubernetes.config.kube_config

                reload(kubernetes.config.kube_config)  # need to refresh environ variables used for kubeconfig
            else:
                raise Exception(f"Unknown Kubernetes Provider {self.k8s_provider}")

        # Save the Tempo pipeline
        conda_env_path = os.path.join(classifier.get_tempo().details.local_folder, "conda.yaml")
        with open(conda_env_path, "w") as f:
            f.write(self.conda_env)
        save(classifier)
        from os import listdir

        classifier_files = [
            (os.path.join("classifier", f), os.path.join(classifier.get_tempo().details.local_folder, f))
            for f in listdir(classifier.get_tempo().details.local_folder)
        ]
        # Store classifier files
        with S3(run=self) as s3:
            s3.put_files(iter(classifier_files))

        # Deploy Tempo pipeline
        if self.run_local:
            print("Deploying locally")
            remote_model = deploy(classifier)
            time.sleep(10)
            print(remote_model.predict(np.array([[1, 2, 3, 4]])))
            remote_model.undeploy()
        else:
            print("Deploying to production")
            from tempo.seldon.k8s import SeldonCoreOptions
            from tempo.serve.metadata import KubernetesOptions

            runtime_options = SeldonCoreOptions(
                k8s_options=KubernetesOptions(namespace="production", authSecretName="s3-secret")
            )

            remote_model = deploy(classifier, options=runtime_options)
            print(remote_model.predict(np.array([[1, 2, 3, 4]])))

        self.next(self.end)

    @step
    def end(self):
        pass


if __name__ == "__main__":
    IrisFlow()
