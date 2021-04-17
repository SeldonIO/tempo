import joblib
from sklearn.linear_model import LogisticRegression
from src.constants import SKLearnFolder, XGBFolder
from src.data import IrisData


def train_lr(artifacts_folder: str, data: IrisData):
    logreg = LogisticRegression(C=1e5)
    logreg.fit(data.X, data.y)
    with open(f"{artifacts_folder}/{SKLearnFolder}/model.joblib", "wb") as f:
        joblib.dump(logreg, f)


def train_xgb(artifacts_folder: str, data: IrisData):
    from xgboost import XGBClassifier

    clf = XGBClassifier()
    clf.fit(data.X, data.y)
    clf.save_model(f"{artifacts_folder}/{XGBFolder}/model.bst")
