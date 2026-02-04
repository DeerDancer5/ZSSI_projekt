import joblib
import pandas as pd
from src.preprocessing.features import add_feature_engineering


def load_model(path="model.pkl"):
    return joblib.load(path)

def prepare_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])

    # feature engineering
    df = add_feature_engineering(df)

    for col in ["Name", "PassengerId", "Cabin", "Ticket"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df

def predict_passenger(model, passenger_data):
    df = prepare_input(passenger_data)
    proba = model.predict_proba(df)[0,1]
    pred = model.predict(df)[0]
    return pred, proba

if __name__ == "__main__":

    print("Loading model...")
    model = load_model()

    passenger = {
        "Pclass": 1,
        "Name": "Smith, Mr. John",
        "Sex": "male",
        "Age": 28,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 72.0,
        "Embarked": "C",
        "PassengerId": 999,
        "Cabin": None,
        "Ticket": "A/5 21171"
    }

    print("Running prediction...")
    pred, proba = predict_passenger(model, passenger)

    print("\n===== RESULT =====")
    print("Prediction (0=did not survive, 1=survived):", pred)
    print("Probability of survival:", round(proba, 4))