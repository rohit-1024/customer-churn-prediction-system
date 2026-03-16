import pandas as pd
import numpy as np


def create_tenure_group(tenure):

    if tenure <= 12:
        return "0-12_months"

    elif tenure <= 24:
        return "12-24_months"

    elif tenure <= 48:
        return "24-48_months"

    else:
        return "48+_months"


def preprocess_input(user_input, training_columns):

    df = pd.DataFrame([user_input])

    # ------------------------------
    # Binary Encoding
    # ------------------------------

    binary_cols = [
        "partner",
        "dependents",
        "phone_service",
        "paperless_billing",
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies",
        "multiple_lines"
    ]

    for col in binary_cols:

        df[col] = df[col].map({
            "Yes": 1,
            "No": 0,
            "No internet service": 0,
            "No phone service": 0
        })

    # Gender encoding
    df["gender"] = df["gender"].map({
        "Male": 1,
        "Female": 0
    })

    # ------------------------------
    # Feature Engineering
    # ------------------------------

    df["tenure_group"] = df["tenure"].apply(create_tenure_group)

    service_columns = [
        "online_security",
        "online_backup",
        "device_protection",
        "tech_support",
        "streaming_tv",
        "streaming_movies"
    ]

    df["services_count"] = df[service_columns].sum(axis=1)

    df["avg_monthly_spend"] = df["total_charges"] / df["tenure"]

    df["avg_monthly_spend"] = df["avg_monthly_spend"].replace([np.inf, -np.inf], 0)

    # ------------------------------
    # One-Hot Encoding
    # ------------------------------

    df = pd.get_dummies(df)

    # ------------------------------
    # Align with training columns
    # ------------------------------

    df = df.reindex(columns=training_columns, fill_value=0)

    return df
