# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load

# -------------------- CONFIG --------------------
DATA_PATH = "digital_marketing_campaign_dataset.csv"
MODEL_PATH = "xgb_6features_best.joblib"
ENCODER_PATH = "gender_encoder.joblib"
THRESHOLD_PATH = "xgb_6features_best_threshold.joblib"

FEATURE_COLS = ["Age", "Gender", "Income", "AdSpend", "WebsiteVisits", "TimeOnSite"]


@st.cache_resource
def load_everything():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=["CustomerID", "AdvertisingPlatform", "AdvertisingTool"], errors="ignore")

    model = load(MODEL_PATH)
    le_gender = load(ENCODER_PATH)
    best_threshold = load(THRESHOLD_PATH)

    return df, model, le_gender, best_threshold


df, xgb_model, le_gender, best_threshold = load_everything()

# -------------------- APP --------------------
st.title(" Customer Conversion Prediction (Tuned 6-Feature XGBoost)")

home_tab, predict_tab = st.tabs(["🏠 Home", "🧩 Predict Conversion"])

# -------------------- HOME --------------------
with home_tab:
    st.header("Welcome 👋")
    st.write("""
    This app uses a **GridSearch-tuned XGBoost model** trained on 6 features:

    - Age  
    - Gender  
    - Income  
    - AdSpend  
    - WebsiteVisits  
    - TimeOnSite  

    The model also learns an **optimal decision threshold** (not always 0.5)  
    to better separate **Converted** and **Not Converted** customers.
    """)

    if "Conversion" in df.columns:
        conv_counts = df["Conversion"].value_counts().sort_index()
        labels = ["Not Converted (0)", "Converted (1)"]
        values = conv_counts.values

        fig, ax = plt.subplots()
        ax.bar(labels, values, color=["red", "green"])
        ax.set_ylabel("Count")
        ax.set_title("Conversion Distribution in Dataset")
        st.pyplot(fig)

        st.write("Counts:", conv_counts.to_dict())

    st.subheader(" Dataset Sample")
    show_cols = [c for c in FEATURE_COLS + ["Conversion"] if c in df.columns]
    st.dataframe(df[show_cols].head())

    st.info("Go to the **🧩 Predict Conversion** tab to try your own values.")


# -------------------- PREDICT --------------------
with predict_tab:
    st.header("Enter Customer Details (6 Features)")

    st.write(f" Model's best threshold (from training): **{best_threshold:.2f}**")
    user_threshold = st.slider(
        "Adjust decision threshold (optional):",
        min_value=0.1,
        max_value=0.9,
        value=float(best_threshold),
        step=0.05
    )

    # Inputs (wide ranges for experimentation)
    age = st.number_input("Age", min_value=10.0, max_value=100.0, value=30.0, step=1.0)

    gender_options = list(le_gender.classes_)
    gender = st.selectbox("Gender", options=gender_options)

    income = st.number_input("Income", min_value=0.0, max_value=1_000_000.0, value=50_000.0, step=1000.0)
    ad_spend = st.number_input("AdSpend", min_value=0.0, max_value=1_000_000.0, value=500.0, step=10.0)
    website_visits = st.number_input("WebsiteVisits", min_value=0.0, max_value=10_000.0, value=20.0, step=1.0)
    time_on_site = st.number_input("TimeOnSite (seconds)", min_value=0.0, max_value=100_000.0, value=600.0, step=10.0)

    # Encode gender exactly like in training
    gender_encoded = le_gender.transform([gender])[0]

    input_df = pd.DataFrame([{
        "Age": age,
        "Gender": gender_encoded,
        "Income": income,
        "AdSpend": ad_spend,
        "WebsiteVisits": website_visits,
        "TimeOnSite": time_on_site
    }], columns=FEATURE_COLS)

    if st.button(" Predict Conversion"):
        proba = xgb_model.predict_proba(input_df)[0, 1]
        pred_class = int(proba >= user_threshold)

        st.subheader(" Input Summary")
        st.table(pd.DataFrame(
            [
                ("Age", age),
                ("Gender", gender),
                ("Gender (encoded)", gender_encoded),
                ("Income", income),
                ("AdSpend", ad_spend),
                ("WebsiteVisits", website_visits),
                ("TimeOnSite", time_on_site),
            ],
            columns=["Feature", "Value"]
        ))

        st.subheader("Prediction Result")
        st.write(f"**Conversion Probability:** `{proba:.3f}`")
        st.write(f"**Used Threshold:** `{user_threshold:.2f}`")

        if pred_class == 1:
            st.success(" This customer is **LIKELY TO CONVERT** (1).")
        else:
            st.warning(" This customer is **UNLIKELY TO CONVERT** (0).")

        st.subheader(" Probability Chart")
        labels = ["Not Convert (%)", "Convert (%)"]
        values = [(1 - proba) * 100, proba * 100]

        fig, ax = plt.subplots()
        ax.pie(values, labels=labels, autopct="%1.1f%%", colors=["red", "green"])
        ax.set_title("Conversion Probability")
        st.pyplot(fig)
