import dill
from alibi.explainers import AnchorTabular
from sklearn.ensemble import RandomForestClassifier
from src.constants import EXPLAINER_FOLDER
from src.data import AdultData


def train_explainer(artifacts_folder: str, data: AdultData, model: RandomForestClassifier) -> AnchorTabular:
    def predict_fn(x):
        return model.predict(x)

    explainer = AnchorTabular(predict_fn, data.feature_names, categorical_names=data.category_map, seed=1)
    explainer.fit(data.X_train, disc_perc=(25, 50, 75))
    with open(f"{artifacts_folder}/{EXPLAINER_FOLDER}" + "/explainer.dill", "wb") as f:
        explainer.predictor = None
        explainer.samplers[0].predictor = None
        dill.dump(explainer, f)
    return explainer
