import pandas as pd
import joblib

model = joblib.load("xgboost_model.pkl")
preprocessor = joblib.load("preprocessor.pkl")

def score_new_data(input_df):
    input_df = input_df.copy()

    cols_to_drop = ["is_canceled", "booking_id", "reservation_status", "reservation_status_date", "agent"]
    input_df = input_df.drop(columns=[col for col in cols_to_drop if col in input_df.columns], errors="ignore")

    processed = preprocessor.transform(input_df)

    if hasattr(processed, "toarray"):
        processed = processed.toarray()

    input_df["predicted_is_canceled"] = model.predict(processed)
    input_df["cancellation_probability"] = model.predict_proba(processed)[:, 1]

    return input_df