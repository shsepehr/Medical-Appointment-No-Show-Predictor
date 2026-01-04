import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from preprocess import preprocess_data
from features import add_features

# Load data
df = pd.read_csv("data/appointments.csv")

# Preprocess & features
df = preprocess_data(df)
df = add_features(df)

X = df.drop(columns=["show", "scheduled_day", "appointment_day"])
y = df["show"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

os.makedirs("model", exist_ok=True)
joblib.dump(pipeline, "model/no_show_model.pkl")

print("Model saved successfully.")
