import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

#load models
mlp_model = load_model('models/mlp_model.keras')
log_model = joblib.load('models/logistic_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')

#Define feature names
feature_names = ['radius1', 'texture1', 'perimeter1', 'area1', 
                  'smoothness1', 'compactness1', 'concavity1', 
                  'concave_points1', 'symmetry1', 'fractal_dimension1', 
                  'radius2', 'texture2', 'perimeter2', 'area2', 
                  'smoothness2', 'compactness2', 'concavity2', 
                  'concave_points2', 'symmetry2', 'fractal_dimension2', 
                  'radius3', 'texture3', 'perimeter3', 'area3', 
                  'smoothness3', 'compactness3', 'concavity3', 
                  'concave_points3', 'symmetry3', 'fractal_dimension3', 
]

#example values for testing
example_values = {
    'radius1': 14.0, 'texture1': 20.0, 'perimeter1': 90.0, 'area1': 600.0,
    'smoothness1': 0.1, 'compactness1': 0.1, 'concavity1': 0.15, 'concave_points1': 0.07,
    'symmetry1': 0.18, 'fractal_dimension1': 0.06,
    
    'radius2': 0.5, 'texture2': 1.0, 'perimeter2': 3.5, 'area2': 40.0,
    'smoothness2': 0.005, 'compactness2': 0.02, 'concavity2': 0.02, 'concave_points2': 0.01,
    'symmetry2': 0.02, 'fractal_dimension2': 0.003,
    
    'radius3': 16.0, 'texture3': 25.0, 'perimeter3': 110.0, 'area3': 800.0,
    'smoothness3': 0.14, 'compactness3': 0.2, 'concavity3': 0.15, 'concave_points3': 0.1,
    'symmetry3': 0.25, 'fractal_dimension3': 0.07
}


#Streamlit UI
st.title('Breast Cancer Diagnosis Predictior')
st.markdown("""
This app allows you to input tumor characteristics to predict whether a tumor is **Benign** or **Malignant** using different machine learning models.
Choose a model below, adjust the input features, and click **Predict Diagnosis** to view results.
""")

#User input
input_features = {}

#Mean features (first 10)
with st.expander("Mean Features (radius1 to fractal_dimension1)"):
    for feature in feature_names[:10]:
        input_features[feature] = st.number_input(feature, value=example_values[feature])

#Standard Error features (10-20)
with st.expander("Standard Error Features (radius2 to fractal_dimension2)"):
    for feature in feature_names[10:20]:
        input_features[feature] = st.number_input(feature, value=example_values[feature])

#Worst Features (20-30)
with st.expander("Worst Features (radius3 to fractal_dimension3)"):
    for feature in feature_names[20:]:
        input_features[feature] = st.number_input(feature, value=example_values[feature])


#Sidebar for model selection
st.sidebar.title("Model Selector")
model_choice = st.selectbox(
    "🧠 Choose a Model for Prediction",
    options=["MLP Neural Network", "Logistic Regression", "XGBoost"],
    index=0
)

#Prediction
if st.button('Predict Diagnosis'):
    input_df = pd.DataFrame([[input_features[feat] for feat in feature_names]], columns = feature_names)
    input_scaled = scaler.transform(input_df)

    # Display results in Streamlit
    st.header("Cancer Diagnosis Prediction Results")

    #Predict based on model choice
    if model_choice == 'MLP Neural Network':
        y_pred_proba = mlp_model.predict(input_scaled).flatten()[0]
        y_pred = (y_pred_proba > 0.5).astype(int)
        st.subheader('MLP Neural Network')
        st.write("**Prediction:**", "Malignant" if y_pred == 1 else "Benign")
        st.write(f"**Probability of Malignancy:** {y_pred_proba:.2%}")


    elif model_choice == 'Logistic Regression':
        y_pred_proba = log_model.predict_proba(input_scaled)[:, 1][0]
        y_pred = (y_pred_proba > 0.5)
        st.subheader('Logistic Regression')
        st.write("**Prediction:**", "Malignant" if y_pred == 1 else "Benign")
        st.write(f"**Probability of Malignancy:** {y_pred_proba:.2%}")
    
    elif model_choice == 'XGBoost':
        y_pred_proba = xgb_model.predict_proba(input_scaled)[:, 1][0]
        y_pred = (y_pred_proba > 0.5)
        st.subheader('XGBoost Classifier')
        st.write("**Prediction:**", "Malignant" if y_pred == 1 else "Benign")
        st.write(f"**Probability of Malignancy:** {y_pred_proba:.2%}")
    
    else:
        st.error("Please select a valid model.")

with st.sidebar:
    st.markdown("## 📝 App Info")
    st.markdown("Built with `Streamlit`, using real-world UCI Breast Cancer data.")
    st.markdown("Predict using a **Neural Network**, **Logistic Regression**, or **XGBoost**.")
