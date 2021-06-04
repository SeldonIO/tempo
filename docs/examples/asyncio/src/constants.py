import os

RootFolder = os.path.dirname(os.path.dirname(__file__))
ArtifactsFolder = os.path.join(RootFolder, "artifacts")

SKLearnFolder = os.path.join(ArtifactsFolder, "sklearn")
XGBoostFolder = os.path.join(ArtifactsFolder, "xgboost")
