# train_model_RF.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Load the data
data=pd.read_csv('data/card_transdata.csv')


# 2. Prepare features and target
X = data.drop('fraud', axis=1)
y = data['fraud']

# 3. Split into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 4. Train Random Forest
model = RandomForestClassifier(class_weight='balanced', random_state=42)
model.fit(X_train, y_train)

# 5. Save model and test set
joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(X_test, 'X_test.pkl')
joblib.dump(y_test, 'y_test.pkl')

print("✅ Model trained and test set saved!")
