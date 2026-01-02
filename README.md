# diabetes-prediction-app
🩺 Diabetes Prediction App — Project Description

📌 Overview
The Diabetes Prediction App is a machine learning–based web application that predicts the risk of diabetes using key medical and demographic parameters. Built using Python, Scikit-learn, and Streamlit, the application provides an interactive interface for both individual risk assessment and batch prediction via CSV upload.

The app is designed for educational and screening purposes, demonstrating how machine learning models can be trained, evaluated, and deployed as real-time web applications.

🎯 Objective
The primary objective of this project is to:
	•	Predict the likelihood of diabetes using clinical input features
	•	Display probability (%) instead of just binary output
	•	Visually highlight risk levels using warning colors
	•	Explain feature importance influencing predictions
	•	Enable easy deployment on Streamlit Community Cloud

🧠 Machine Learning Model
	•	Algorithm Used: Logistic Regression
	•	Dataset: PIMA Indians Diabetes Dataset
	•	Target Variable: Outcome (0 = Non-Diabetic, 1 = Diabetic)

The model is trained after preprocessing the dataset and splitting it into training and testing sets. Logistic Regression is chosen due to its interpretability and effectiveness for binary classification problems in healthcare.


🧪 Input Features
The prediction is based on the following parameters:
	•	Pregnancies
	•	Glucose Level
	•	Blood Pressure
	•	Skin Thickness
	•	Insulin
	•	Body Mass Index (BMI)
	•	Diabetes Pedigree Function (genetic risk)
	•	Age

These features are commonly used in clinical diabetes risk assessment.



🚦 Output & Risk Interpretation
Instead of only showing Diabetic / Non-Diabetic, the app displays:
	•	Probability (%) of diabetes
	•	Color-coded risk levels:
	•	🟢 Low Risk (<30%)
	•	🟠 Medium Risk (30–60%)
	•	🔴 High Risk (>60%)

This makes the results more intuitive and user-friendly.



📊 Feature Importance
The application visualizes feature importance using model coefficients, helping users understand which medical factors most influence the prediction. This improves transparency and interpretability of the model.


📁 CSV Upload (Batch Prediction)
Users can upload a CSV file containing multiple records.
The app processes all entries and outputs:
	•	Diabetes risk percentage
	•	Final prediction for each record

This feature enables bulk analysis, making the app scalable beyond single-user input.

🌐 Deployment

The application is deployed using Streamlit Community Cloud, making it accessible through a public web link.
Deployment is handled via a GitHub repository, ensuring version control and reproducibility.

⚠️ Disclaimer

This application is intended only for educational and demonstration purposes.
It does not replace professional medical diagnosis or advice.
