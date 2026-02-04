from sklearn.model_selection import train_test_split

from src.modeling.feature_importance import visualize_feature_importance
from src.utils.io import load_titanic
from src.preprocessing.features import add_feature_engineering
from src.modeling.pipeline import build_full_pipeline
from src.modeling.tuning import build_grid_search
from src.modeling.evaluation import plot_confusion, plot_roc
import joblib


def main():

    print("Loading data...")
    df = load_titanic("data/raw/titanic.csv")

    # Drop useless columns
    df = df.drop(columns=["Cabin", "Ticket"])

    print("Adding engineered features...")
    df = add_feature_engineering(df)

    # Features / label
    X = df.drop(columns=["Survived", "Name", "PassengerId"])
    y = df["Survived"]

    # ===== SPLIT =====
    print("Splitting data into train and test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    print("Building pipeline...")
    pipeline = build_full_pipeline()

    print("Configuring GridSearchCV...")
    grid = build_grid_search(pipeline)

    print("Training model (GridSearchCV on TRAIN only)...")
    grid.fit(X_train, y_train)

    print("\n===== BEST PARAMS =====")
    print(grid.best_params_)

    print("\n===== BEST CV SCORE (ROC-AUC on TRAIN folds) =====")
    print(grid.best_score_)

    # ===== FINAL TEST EVALUATION =====
    print("\nEvaluating best model on TEST data...")
    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    plot_confusion(y_test, y_pred)
    plot_roc(y_test, y_proba)

    visualize_feature_importance(best_model, X_train)

    joblib.dump(best_model, "model.pkl")
    print("Saved best model to model.pkl")

    print("\nDone!")

if __name__ == "__main__":
    main()