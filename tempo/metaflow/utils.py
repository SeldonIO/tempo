from typing import Any

from metaflow import S3, FlowSpec, IncludeFile, Step
from metaflow.plugins.aws.batch.batch_decorator import BatchDecorator

from tempo.serve.metadata import ModelFramework
from tempo.serve.model import Model


def save_artifact(model: Any, filename: str):
    """

    Parameters
    ----------
    model The model bytes to save.
    filename The filename for the model

    Returns
    -------
    Return the local folder

    """
    import os
    import tempfile

    folder = tempfile.mkdtemp()
    print(folder)
    local_model_path = os.path.join(folder, filename)
    with open(local_model_path, "wb") as f:
        f.write(model)
    return folder


def gke_authenticate(kubeconfig: IncludeFile, gsa_key: IncludeFile):
    """
    Authenticate to GKE cluster as shown in
    https://cloud.google.com/kubernetes-engine/docs/how-to/api-server-authentication#environments-without-gcloud
    Parameters
    ----------
    kubeconfig A Metaflow IncludeFile for the GKE Kubeconfig
    gsa_key A Metaflow IncludeFile for the gsa_key

    """
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


def aws_authenticate(eks_cluster_name: str):
    import os
    import tempfile
    from importlib import reload

    import kubernetes.config.kube_config

    k8s_folder = tempfile.mkdtemp()

    # install AWS IAM authenticator for k8s
    os.system(
        (
            "curl -o /usr/local/bin/aws-iam-authenticator "
            "https://amazon-eks.s3-us-west-2.amazonaws.com/1.21.2/2021-07-05/bin/linux/"
            "amd64/aws-iam-authenticator && chmod +x /usr/local/bin/aws-iam-authenticator"
        )
    )
    kubeconfig_path = os.path.join(k8s_folder, "kubeconfig.yaml")
    with open(kubeconfig_path, "w") as f:
        f.write(generate_eks_config(eks_cluster_name))
    print(generate_eks_config(eks_cluster_name))
    os.environ["KUBECONFIG"] = kubeconfig_path
    reload(kubernetes.config.kube_config)


def generate_eks_config(cluster_name: str) -> str:
    """ Use AWS API to generate kubeconfig given EKS cluster name """
    import boto3

    eks = boto3.client("eks")

    # Get cluster details from EKS API
    resp = eks.describe_cluster(name=cluster_name)

    endpoint = resp["cluster"]["endpoint"]
    ca_cert = resp["cluster"]["certificateAuthority"]["data"]

    return f"""apiVersion: v1
clusters:
- cluster:
    server: {endpoint}
    certificate-authority-data: {ca_cert}
  name: kubernetes
contexts:
- context:
    cluster: kubernetes
    user: aws
  name: aws
current-context: aws
kind: Config
preferences: {{}}
users:
- name: aws
  user:
    exec:
      apiVersion: client.authentication.k8s.io/v1alpha1
      command: aws-iam-authenticator
      args:
        - "token"
        - "-i"
        - "{cluster_name}"
    """


def create_s3_folder(flow_spec: FlowSpec, folder_name: str) -> str:
    """
    Create an S3 folder within the Flow
    Parameters
    ----------
    flow_spec The running Flow
    folder_name The folder name

    Returns
    -------
    Path S3 path

    """
    import os

    from metaflow import S3

    try:
        with S3(run=flow_spec) as s3:
            return os.path.split(s3.put(folder_name + "/.keep", "keep"))[0]
    except TypeError:
        return ""


def upload_s3_folder(flow_spec: FlowSpec, s3_folder_name: str, local_path: str):
    """
    Upload file in a local path to Flow S3 Folder
    Parameters
    ----------
    flow_spec The running Flow
    s3_folder_name The S3 destination folder name
    local_path The local path to search for files

    """
    import os

    artifact_files = [(os.path.join(s3_folder_name, f), os.path.join(local_path, f)) for f in os.listdir(local_path)]
    with S3(run=flow_spec) as s3:
        s3.put_files(iter(artifact_files))


def save_pipeline_with_conda(pipeline, folder: str, conda_env: IncludeFile):
    """
    Save a Tempo pipeline using a Conda enviroment from a Metaflow IncludeFile
    Parameters
    ----------
    pipeline The Tempo pipeline
    folder The local folder to save Tempo pipeline to
    conda_env The conda environment as a Metaflow IncludeFile

    """
    import os

    from tempo import save

    conda_env_path = os.path.join(folder, "conda.yaml")
    with open(conda_env_path, "w") as f:
        f.write(conda_env)
    save(pipeline)


def running_aws_batch(step: Step) -> bool:
    """
    Test if a Step is running on AWS batch
    Parameters
    ----------
    step The step to test

    Returns
    -------
    True if flow is running on AWS Batch

    """
    running_on_aws_batch = False
    for deco in step.decorators:  # pylint: disable=maybe-no-member
        if isinstance(deco, BatchDecorator):
            running_on_aws_batch = True
    return running_on_aws_batch


def create_sklearn_model(artifact: Any, flow_spec: FlowSpec) -> Model:
    """
    Save and upload to flow S3 a Tempo SKLearn model

    Parameters
    ----------
    artifact: SKLearn artifact
    flow_spec: running Flow

    Returns
    -------
    Tempo SKLearn model

    """
    sklearn_local_path = save_artifact(artifact, "model.joblib")
    sklearn_url = create_s3_folder(flow_spec, "sklearn")
    sklearn_model = Model(
        name="test-iris-sklearn",
        platform=ModelFramework.SKLearn,
        local_folder=sklearn_local_path,
        uri=sklearn_url,
        description="An SKLearn Iris classification model",
    )
    if sklearn_url:
        upload_s3_folder(flow_spec, "sklearn", sklearn_local_path)
    return sklearn_model


def create_xgboost_model(artifact: Any, flow_spec: FlowSpec) -> Model:
    """
    Save and upload to flow S3 a Tempo XGBoost model

    Parameters
    ----------
    artifact: XGBost artifact
    flow_spec: running Flow

    Returns
    -------
    Tempo XGBoost model

    """
    xgboost_local_path = save_artifact(artifact, "model.bst")
    xgboost_url = create_s3_folder(flow_spec, "xgboost")
    xgboost_model = Model(
        name="test-iris-xgboost",
        platform=ModelFramework.XGBoost,
        local_folder=xgboost_local_path,
        uri=xgboost_url,
        description="An XGBoost Iris classification model",
    )
    if xgboost_url:
        upload_s3_folder(flow_spec, "xgboost", xgboost_local_path)
    return xgboost_model
