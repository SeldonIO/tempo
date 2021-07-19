from metaflow import FlowSpec, step, Task, IncludeFile, conda
from typing import Any
import functools

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
                print('Pip Install:', library, version)
                subprocess.run([sys.executable, '-m', 'pip', 'install', '--quiet', library + '==' + version])
            return function(*args, **kwargs)

        return wrapper

    return decorator

def save_bytes_local(model: Any, model_name: str):
    import tempfile
    import shutil
    import os

    folder = tempfile.mkdtemp()
    print(folder)
    local_model_path = os.path.join(folder, model_name)
    with open(local_model_path, 'wb') as f:
        shutil.copyfileobj(model, f)
    return folder

class IrisFlow(FlowSpec):
    """


    The flow performs the following steps:

    1) Load Iris Data
    """

    conda_env = IncludeFile("conda_env",
                             help="The path to conda environment for classifier",
                             default=script_path('conda.yaml'))

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
        from sklearn.linear_model import LogisticRegression
        from io import BytesIO
        from joblib import dump

        lr = LogisticRegression(C=1e5)
        lr.fit(self.X, self.y)
        dump(lr, script_path("model.joblib"))
        with open(script_path('model.joblib'), 'rb') as fh:
            self.buffered_lr_model = BytesIO(fh.read())


        self.next(self.join)

    @conda(libraries={"xgboost": "1.4.0"})
    @step
    def train_xgboost(self):
        from xgboost import XGBClassifier
        from io import BytesIO

        xgb = XGBClassifier()
        xgb.fit(self.X, self.y)
        xgb.save_model(script_path("model.bst"))
        with open(script_path('model.bst'), 'rb') as fh:
            self.buffered_xgb_model = BytesIO(fh.read())
        self.next(self.join)

    @step
    def join(self, inputs):
        self.merge_artifacts(inputs)

        self.next(self.tempo)

    @conda(libraries={"numpy":"1.19.5"})
    @pip(libraries={'mlops-tempo': '0.2.0'})
    @step
    def tempo(self):
        from deploy import get_tempo_artifacts
        from tempo.serve.loader import save
        from tempo import deploy
        import os
        import time
        import numpy as np

        local_sklearn_path = save_bytes_local(self.buffered_lr_model, "model.joblib")
        local_xgb_path = save_bytes_local(self.buffered_xgb_model, "model.bst")
        classifier, sklearn_model, xgboost_model = get_tempo_artifacts(local_sklearn_path, local_xgb_path)

        print(self.conda_env)
        conda_env_path = os.path.join(classifier.get_tempo().details.local_folder,"conda.yaml")
        print(conda_env_path)
        with open(conda_env_path,"w") as f:
            f.write(self.conda_env)

        save(classifier)
        remote_model = deploy(classifier)
        time.sleep(10)
        print(remote_model.predict(np.array([[1, 2, 3, 4]])))
        remote_model.undeploy()


        self.next(self.end)

    @step
    def end(self):
        pass

if __name__ == '__main__':
    IrisFlow()
