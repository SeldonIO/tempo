from typing import Any

from metaflow import S3, FlowSpec, IncludeFile


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

    with S3(run=flow_spec) as s3:
        return os.path.split(s3.put(folder_name + "/.keep", "keep"))[0]


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
