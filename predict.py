import joblib
import pandas as pd

model = joblib.load("model/no_show_model.pkl")

sample = pd.DataFrame([{
    "age": 40,
    "gender": 1,
    "sms_received": 1,
    "previous_no_show": 0,
    "waiting_days": 5,
    "age_group": 4
}])

prediction = model.predict(sample)

print("No-Show Prediction:", "Yes" if prediction[0] == 1 else "No")
