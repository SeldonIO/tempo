from metaflow import FlowSpec, IncludeFile, Parameter, conda, step
from utils import pip

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
    kubeconfig = IncludeFile("kubeconfig", help="The path to kubeconfig", default=script_path("kubeconfig.yaml"))
    gsa_key = IncludeFile(
        "gsa_key", help="The path to google service account json", default=script_path("gsa-key.json")
    )
    k8s_provider = Parameter(
        "k8s_provider", help="kubernetes provider. Needed for non local run to deploy", default="gke"
    )
    eks_cluster_name = Parameter("eks_cluster_name", help="AWS EKS cluster name (if using EKS)", default="")

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

    def create_tempo_artifacts(self):
        import tempfile

        from deploy import get_tempo_artifacts

        from tempo.metaflow.utils import create_s3_folder, save_artifact, save_pipeline_with_conda, upload_s3_folder

        # Store models to local artifact locations
        local_sklearn_path = save_artifact(self.buffered_lr_model, "model.joblib")
        local_xgb_path = save_artifact(self.buffered_xgb_model, "model.bst")
        local_pipeline_path = tempfile.mkdtemp()
        # Create S3 folders for artifacts
        classifier_url = create_s3_folder(self, PIPELINE_FOLDER_NAME)
        sklearn_url = create_s3_folder(self, SKLEARN_FOLDER_NAME)
        xgboost_url = create_s3_folder(self, XGBOOST_FOLDER_NAME)

        classifier, sklearn_model, xgboost_model = get_tempo_artifacts(
            local_sklearn_path, local_xgb_path, local_pipeline_path, sklearn_url, xgboost_url, classifier_url
        )
        # Create pipeline artifacts
        save_pipeline_with_conda(classifier, local_pipeline_path, self.conda_env)
        if classifier_url:  # Check running with S3 access
            # Upload artifacts to S3
            upload_s3_folder(self, PIPELINE_FOLDER_NAME, local_pipeline_path)
            upload_s3_folder(self, SKLEARN_FOLDER_NAME, local_sklearn_path)
            upload_s3_folder(self, XGBOOST_FOLDER_NAME, local_xgb_path)
            return classifier, True
        else:
            return classifier, False

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
        from tempo.metaflow.utils import aws_authenticate, gke_authenticate
        from tempo.serve.deploy import get_client
        from tempo.serve.metadata import SeldonCoreOptions

        if self.k8s_provider == "gke":
            gke_authenticate(self.kubeconfig, self.gsa_key)
        elif self.k8s_provider == "aws":
            aws_authenticate(self.eks_cluster_name)
        else:
            raise Exception(f"Unknown Kubernetes Provider {self.k8s_provider}")

        runtime_options = SeldonCoreOptions(
            **{"remote_options": {"namespace": "production", "authSecretName": "s3-secret"}}
        )

        remote_model = deploy_remote(classifier, options=runtime_options)
        self.client_model = get_client(remote_model)
        time.sleep(10)
        print(self.client_model.predict(np.array([[1, 2, 3, 4]])))

    @conda(libraries={"numpy": "1.19.5"})
    @pip(libraries={"mlops-tempo": "0.5.0", "conda_env": "2.4.2"})
    @step
    def tempo(self):
        """
        Create Tempo artifacts locally and saved to S3 within the workflow bucket.
        Then either deploy locally to Docker or deploy to a remote Kubernetes cluster based on the
        --tempo-on-docker parameter
        """
        classifier, remote = self.create_tempo_artifacts()

        if remote:
            self.deploy_tempo_remote(classifier)
        else:
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
