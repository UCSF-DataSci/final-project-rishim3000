#attempt to suppress GPU-related warnings (only kinda works)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

#Necessary packages
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import joblib
from data_preprocessing import preprocess_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time
import seaborn as sns

breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17)

X = breast_cancer_wisconsin_diagnostic.data.features
y = breast_cancer_wisconsin_diagnostic.data.targets

X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X, y)

#load trained models
mlp_model = load_model('models/mlp_model.keras')
log_model = joblib.load('models/logistic_model.pkl')
xgb_model = joblib.load('models/xgb_model.pkl')
scaler = joblib.load('models/scaler.pkl')

#Gaussian noise
def add_noise(X, noise_level = 0.1):
    noise = np.random.normal(0, noise_level, X.shape)
    X_noisy = X + noise
    return X_noisy

#stress test in loop
noise_levels = [0, 0.2, 0.4, 0.6, 0.8]

noise_results = []

for noise in noise_levels:
    print(f"\n--- Noise level: {noise} ---")

    X_test_noisy = add_noise(X_test_scaled, noise_level = noise)

    #MLP
    y_pred_mlp_proba = mlp_model.predict(X_test_noisy)
    y_pred_mlp = (y_pred_mlp_proba > 0.5).astype(int).flatten()
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    noise_results.append(({'noise_level': noise, 'model_name': 'MLP', 'accuracy': acc_mlp}))

    #Logistic Regression
    y_pred_log = log_model.predict(X_test_noisy)
    acc_log = accuracy_score(y_test, y_pred_log)
    noise_results.append(({'noise_level': noise, 'model_name': 'Logistic Regression', 'accuracy': acc_log}))

    #XGBoost
    y_pred_xgb = xgb_model.predict(X_test_noisy)
    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    noise_results.append(({'noise_level': noise, 'model_name': 'XGBoost', 'accuracy': acc_xgb}))

df_noise = pd.DataFrame(noise_results)

print(df_noise)

sns.lineplot(data = df_noise,  x = 'noise_level', y = 'accuracy', hue = 'model_name', marker = 'o')
plt.title('Model Robustness to Noise')
plt.ylim(0.75, 1.0)
plt.grid(True)
plt.savefig('Noise_robustness.png')
plt.show()

#Outliers


