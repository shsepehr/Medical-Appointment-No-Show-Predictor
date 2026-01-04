import pandas as pd

def preprocess_data(df):
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"])
    df["appointment_day"] = pd.to_datetime(df["appointment_day"])

    df["waiting_days"] = (df["appointment_day"] - df["scheduled_day"]).dt.days
    df["gender"] = df["gender"].map({"M": 0, "F": 1})
    df["show"] = df["show"].map({"Yes": 0, "No": 1})

    return df
