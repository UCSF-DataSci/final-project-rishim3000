#Adding src/ to path
import sys
import os
sys.path.append(os.path.abspath("./src")) 


#Load previous functions
from data_preprocessing import preprocess_data
from models import build_mlp, train_logistic, train_xgboost

#packages
from sklearn.metrics import accuracy_score
import pandas as pd
from ucimlrepo import fetch_ucirepo 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
  
# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets

#preprocess
X_train_scaled, X_test_scaled, y_train, y_test, scaler = preprocess_data(X,y)

#Training
mlp_model = build_mlp(X_train_scaled.shape[1])

#Callbacks
early_stopping = EarlyStopping(
    monitor = 'val_loss',
    patience = 5,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor = 'val_loss',
    factor = 0.5,
    patience = 3,
    verbose = 1
)
mlp_model.fit(X_train_scaled, y_train, epochs = 50, batch_size = 32, callbacks = [early_stopping, lr_scheduler], verbose = 0)

log_model = train_logistic(X_train_scaled, y_train)
xgb_model = train_xgboost(X_train_scaled, y_train)

#Evaluate
mlp_preds = (mlp_model.predict(X_test_scaled) > 0.5).astype(int).ravel()
log_preds = log_model.predict(X_test_scaled)
xgb_preds = xgb_model.predict(X_test_scaled)

print('MLP Accuracy:', accuracy_score(y_test, mlp_preds))
print('Logistic Regression Accuracy:', accuracy_score(y_test, log_preds))
print('XGBoost Accuracy:', accuracy_score(y_test, xgb_preds))



#All 3 models have similarly good accuracy > 0.97

#save models
mlp_model.save('models/mlp_model.keras')
joblib.dump(log_model, 'models/logistic_model.pkl')
joblib.dump(xgb_model, 'models/xgb_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')



