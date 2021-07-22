from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.constants import MODEL_FOLDER
from src.data import AdultData


def train_model(artifacts_folder: str, data: AdultData):
    ordinal_features = [x for x in range(len(data.feature_names)) if x not in list(data.category_map.keys())]
    ordinal_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_features = list(data.category_map.keys())
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", ordinal_transformer, ordinal_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    clf = RandomForestClassifier(n_estimators=50)
    model = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    model.fit(data.X_train, data.Y_train)

    print("Train accuracy: ", accuracy_score(data.Y_train, model.predict(data.X_train)))
    print("Test accuracy: ", accuracy_score(data.Y_test, model.predict(data.X_test)))

    dump(model, f"{artifacts_folder}/{MODEL_FOLDER}" + "/model.joblib")
    return model
