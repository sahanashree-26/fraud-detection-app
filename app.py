
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.title("Fraud Detection System with Smart Freeze")

# Load the dataset
uploaded_file = st.file_uploader("Upload the official fraud CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Feature Engineering
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    df['PreviousTransactionDate'] = pd.to_datetime(df['PreviousTransactionDate'])
    df['Time Gap'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds()
    df['Amount/Balance'] = df['TransactionAmount'] / (df['AccountBalance'] + 1)
    df['High Login Attempts'] = df['LoginAttempts'].apply(lambda x: 1 if x > 3 else 0)

    # Fill missing values if any
    df.fillna(0, inplace=True)

    # AI Model: Random Forest
    features = ['TransactionAmount', 'Time Gap', 'Amount/Balance', 'High Login Attempts']
    if 'is_fraud' not in df.columns:
        st.error("❌ 'is_fraud' column not found in the dataset. Please use the original dataset.")
    else:
        X = df[features]
        y = df['is_fraud']
        model = RandomForestClassifier()
        model.fit(X, y)

        df['is_fraud_prediction'] = model.predict(X)
        df['Frozen Features'] = df['is_fraud_prediction'].apply(lambda x: ['Transfer Money', 'Change Password'] if x == 1 else [])

        # Display predictions
        st.subheader("Fraud Prediction Results")
        st.write(df[['TransactionID', 'is_fraud_prediction', 'Frozen Features']])

        # Alert system
        if any(df['is_fraud_prediction'] == 1):
            st.error("ALERT: Fraudulent transaction(s) detected! App features frozen.")
        else:
            st.success("✅ No fraud detected.")
