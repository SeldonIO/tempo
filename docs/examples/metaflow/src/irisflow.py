from metaflow import FlowSpec, IncludeFile, Parameter, conda, step
from utils import pip
import os


PIPELINE_FOLDER_NAME = "classifier"
SKLEARN_FOLDER_NAME = "sklearn"
XGBOOST_FOLDER_NAME = "xgboost"


def script_path(filename):
    import os

    filepath = os.path.join(os.path.dirname(__file__))
    return os.path.join(filepath, filename)


class IrisFlow(FlowSpec):
    """
    A Flow to train two Iris dataset models and combine them for inference with Tempo

    The flow performs the following steps:

    1) Load Iris Data
    2) Train SKLearn LR Model
    3) Train XGBoost LR Model
    4) Create and deploy Tempo artifacts
    """

    conda_env = IncludeFile(
        "conda_env", help="The path to conda environment for classifier", default=script_path("conda.yaml")
    )

    @conda(libraries={"scikit-learn": "0.24.1"})
    @step
    def start(self):
        """
        Download Iris classification datatset
        """
        # pylint: disable=no-member
        from sklearn import datasets

        iris = datasets.load_iris()
        self.X = iris.data
        self.y = iris.target
        self.next(self.train_sklearn, self.train_xgboost)

    @conda(libraries={"scikit-learn": "0.24.1"})
    @step
    def train_sklearn(self):
        """
        Train a SKLearn Logistic Regression Classifier on dataset and save model as artifact
        """
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
        """
        Train an XGBoost classifier on the dataset and save model as artifact
        """
        from xgboost import XGBClassifier

        xgb = XGBClassifier()
        xgb.fit(self.X, self.y)
        xgb.save_model(script_path("model.bst"))
        with open(script_path("model.bst"), "rb") as fh:
            self.buffered_xgb_model = fh.read()
        self.next(self.join)

    @step
    def join(self, inputs):
        """
        Merge two training runs.
        """
        self.merge_artifacts(inputs)

        self.next(self.tempo)

    def deploy_tempo_local(self, classifier):
        import time

        import numpy as np

        from tempo import deploy_local
        from tempo.serve.deploy import get_client

        remote_model = deploy_local(classifier)
        self.client_model = get_client(remote_model)
        time.sleep(10)
        print(self.client_model.predict(np.array([[1, 2, 3, 4]])))

    def deploy_tempo_remote(self, classifier):
        import time

        import numpy as np

        from tempo import deploy_remote
        from tempo.serve.deploy import get_client
        from tempo.serve.metadata import SeldonCoreOptions

        runtime_options = SeldonCoreOptions(
            **{"remote_options": {"namespace": "production", "authSecretName": "s3-secret"}}
        )

        remote_model = deploy_remote(classifier, options=runtime_options)
        self.client_model = get_client(remote_model)
        time.sleep(10)
        print(self.client_model.predict(np.array([[1, 2, 3, 4]])))

    def create_tempo_artifacts(self):
        from deploy import get_tempo_artifacts

        from tempo.metaflow.utils import create_sklearn_model, create_xgboost_model

        sklearn_model = create_sklearn_model(self.buffered_lr_model, self)
        xgboost_model = create_xgboost_model(self.buffered_xgb_model, self)

        classifier, remote_s3 = get_tempo_artifacts(self, sklearn_model, xgboost_model, self.conda_env)

        return classifier, remote_s3

    @conda(libraries={"numpy": "1.19.5"})
    @pip(libraries={"mlops-tempo": "0.5.2", "conda_env": "2.4.2"})
    @step
    def tempo(self):
        """
        Create Tempo artifacts locally and saved to S3 within the workflow bucket.
        Then either deploy locally to Docker or deploy to a remote Kubernetes cluster based on the
        --tempo-on-docker parameter
        """
        from tempo.metaflow.utils import running_aws_batch

        classifier, s3_active = self.create_tempo_artifacts()
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            print("Deploying to k8s cluster")
            self.deploy_tempo_remote(classifier)
        else:
            print("Deploying to local Docker")
            self.deploy_tempo_local(classifier)

        self.next(self.end)

    @step
    def end(self):
        """
        End flow.
        """
        pass


if __name__ == "__main__":
    IrisFlow()
