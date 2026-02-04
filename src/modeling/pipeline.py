from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from src.preprocessing.transformers import build_column_transformer
from src.preprocessing.selection import build_feature_selector


def build_full_pipeline(k_features="all", mlp_params=None):

    preprocessor, _, _ = build_column_transformer()
    selector = build_feature_selector(k=k_features)

    mlp = MLPClassifier(**(mlp_params or {}))

    pipe = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("select", selector),
            ("mlp", mlp)
        ]
    )

    return pipe