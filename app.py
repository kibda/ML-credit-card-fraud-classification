# app.py

import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

# Load models and test data
model_rf = joblib.load('fraud_detection_model.pkl')
X_test_rf = joblib.load('X_test.pkl')
y_test_rf = joblib.load('y_test.pkl')

model_xgb = joblib.load('fraud_detection_model_XGB.pkl')
X_test_xgb = joblib.load('X_test_XGB.pkl')
y_test_xgb = joblib.load('y_test_XGB.pkl')

model_lr = joblib.load('fraud_detection_model_LR.pkl')
X_test_lr = joblib.load('X_test_LR.pkl')
y_test_lr = joblib.load('y_test_LR.pkl')

# Evaluate models
accuracy_rf = accuracy_score(y_test_rf, model_rf.predict(X_test_rf)) * 100
accuracy_xgb = accuracy_score(y_test_xgb, model_xgb.predict(X_test_xgb)) * 100
accuracy_lr = accuracy_score(y_test_lr, model_lr.predict(X_test_lr)) * 100

# Streamlit app
st.title('🔍 Fraud Detection App')

st.sidebar.header('📄 Input New Transaction Data')

# Sidebar with valid value ranges
distance_from_home = st.sidebar.slider('Distance from Home (km)', 0.0, 1000.0, 10.0)
distance_from_last_transaction = st.sidebar.slider('Distance from Last Transaction (km)', 0.0, 500.0, 1.0)
ratio_to_median_purchase_price = st.sidebar.slider('Ratio to Median Purchase Price', 0.0, 10.0, 1.0)
repeat_retailer = st.sidebar.selectbox('Repeat Retailer?', [0.0, 1.0])
used_chip = st.sidebar.selectbox('Used Chip?', [0.0, 1.0])
used_pin_number = st.sidebar.selectbox('Used PIN Number?', [0.0, 1.0])
online_order = st.sidebar.selectbox('Online Order?', [0.0, 1.0])

# Prepare input DataFrame
new_data = pd.DataFrame({
    'distance_from_home': [distance_from_home],
    'distance_from_last_transaction': [distance_from_last_transaction],
    'ratio_to_median_purchase_price': [ratio_to_median_purchase_price],
    'repeat_retailer': [repeat_retailer],
    'used_chip': [used_chip],
    'used_pin_number': [used_pin_number],
    'online_order': [online_order]
})

# Show model accuracies
st.header('📊 Model Accuracies on Test Sets')
st.write(f"**Random Forest Accuracy:** {accuracy_rf:.2f}%")
st.write(f"**XGBoost Accuracy:** {accuracy_xgb:.2f}%")
st.write(f"**Logistic Regression Accuracy:** {accuracy_lr:.2f}%")

# Choose model for prediction
st.header('🤔 Predict New Transaction')

model_choice = st.selectbox(
    "Choose a model for prediction:",
    ("Random Forest", "XGBoost", "Logistic Regression")
)

if st.button('Predict'):
    if model_choice == "Random Forest":
        prediction = model_rf.predict(new_data)[0]
    elif model_choice == "XGBoost":
        prediction = model_xgb.predict(new_data)[0]
    else:
        prediction = model_lr.predict(new_data)[0]

    if prediction == 1:
        st.error('❗ Fraudulent Transaction Detected ❗')
    else:
        st.success('✅ Legitimate Transaction ✅')
