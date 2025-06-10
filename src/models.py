
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


#MLP function
def build_mlp(input_dim):
    model = Sequential([
        Dense(16, input_shape = (input_dim,), activation = 'relu'),
        Dropout(0.2),
        Dense(8, activation = 'relu'),
        Dense(1, activation = 'sigmoid') #Output for binary classification 
    ])
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model

#Logistic regression
def train_logistic(X_train_scaled, y_train):
    model = LogisticRegression(max_iter = 1000)
    model.fit(X_train_scaled, y_train)
    return model

#XGBoost model
def train_xgboost(X_train_scaled, y_train):
    model = XGBClassifier(use_label_encoder = False, eval_metric = 'logloss')
    model.fit(X_train_scaled, y_train)
    return model



