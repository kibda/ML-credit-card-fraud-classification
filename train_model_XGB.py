# train_model_XGB.py

import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# 1. Load the data
data=pd.read_csv('data/card_transdata.csv')

# 2. Prepare features and target
X = data.drop('fraud', axis=1)
y = data['fraud']

# 3. Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train XGBoost model
model = xgb.XGBClassifier(scale_pos_weight=1, use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

# 5. Save model and test set
joblib.dump(model, 'fraud_detection_model_XGB.pkl')
joblib.dump(X_test, 'X_test_XGB.pkl')
joblib.dump(y_test, 'y_test_XGB.pkl')

print("✅ XGBoost model trained and test set saved!")
