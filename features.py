def add_features(df):
    df["age_group"] = df["age"] // 10
    return df
