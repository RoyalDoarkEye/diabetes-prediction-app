import streamlit as st
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# -------------------------
# Load dataset safely
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "diabetes.csv")

data = pd.read_csv(DATA_PATH)

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# -------------------------
# Streamlit UI
# -------------------------
st.title("🩺 Diabetes Prediction App")
st.write("Predict diabetes risk using machine learning (educational use only).")

st.divider()

# -------------------------
# Manual Input Section
# -------------------------
st.subheader("🔢 Manual Input")

input_data = []
for col in X.columns:
    value = st.number_input(f"{col}", min_value=0.0, step=0.1)
    input_data.append(value)

if st.button("Predict"):
    probability = model.predict_proba([input_data])[0][1] * 100
    prediction = model.predict([input_data])[0]

    if probability < 30:
        st.success(f"🟢 Low Risk — Probability: {probability:.2f}%")
    elif probability < 60:
        st.warning(f"🟠 Medium Risk — Probability: {probability:.2f}%")
    else:
        st.error(f"🔴 High Risk — Probability: {probability:.2f}%")

st.divider()

# -------------------------
# Feature Importance
# -------------------------
st.subheader("📊 Feature Importance")

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": np.abs(model.coef_[0])
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance.set_index("Feature"))

st.divider()

# -------------------------
# CSV Upload Section
# -------------------------
st.subheader("📁 Upload CSV for Batch Prediction")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    uploaded_data = pd.read_csv(uploaded_file)

    if set(X.columns).issubset(uploaded_data.columns):
        probs = model.predict_proba(uploaded_data[X.columns])[:, 1] * 100
        uploaded_data["Diabetes_Risk_%"] = probs
        uploaded_data["Prediction"] = np.where(probs >= 50, "Diabetic", "Non-Diabetic")

        st.success("Prediction completed!")
        st.dataframe(uploaded_data)
    else:
        st.error("CSV must contain the same columns as training data.")

st.divider()

# -------------------------
# Model Performance
# -------------------------
st.subheader("📈 Model Performance")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.write(f"Accuracy: **{accuracy:.2f}**")
st.text("Classification Report:")
st.text(classification_report(y_test, y_pred))
