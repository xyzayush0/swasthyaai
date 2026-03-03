import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer

st.title("🫀 SwasthyaAI - AI Health Risk Predictor")

# Load dataset
dataset = load_breast_cancer()
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data["target"] = dataset.target

# Use first 3 features
X = data.iloc[:, :3]
y = data["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

st.write(f"Model Accuracy: {round(accuracy*100,2)}%")

st.header("Enter Health Parameters")

f1 = st.number_input("Feature 1", value=10.0)
f2 = st.number_input("Feature 2", value=10.0)
f3 = st.number_input("Feature 3", value=10.0)

if st.button("Predict Risk"):
    input_data = np.array([[f1, f2, f3]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("Low Risk")
    else:
        st.error("High Risk")
