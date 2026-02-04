from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_column_transformer():

    numeric_features = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
    categorical_features = ["Sex", "Embarked", "Pclass", "Title"]

    numeric_transformer = SimpleImputer(strategy="median")

    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor, numeric_features, categorical_features