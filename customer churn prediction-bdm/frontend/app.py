import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from io import StringIO

try:
    churn_model = joblib.load("../backend/models/final_churn_model_catboost.joblib")
    clv_model = joblib.load("../backend/models/final_clv_model_cat.joblib")
except FileNotFoundError:
    st.error("Model files not found. Please ensure 'final_churn_model_catboost.joblib' and 'final_clv_model_cat.joblib' are in the directory.")
    st.stop()

numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'ChargesPerTenure', 'ServiceCount', 'ARPU', 'RevenuePotential', 'ContractLength', 'AvgChargesPerService', 'ServicesCount', 'RetentionScore']
categorical_features = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TenureGroup', 'AutoPay', 'SeniorFlag']

if 'df' not in st.session_state:
    st.session_state.df = None
if 'processed_df' not in st.session_state:
    st.session_state.processed_df = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'segments' not in st.session_state:
    st.session_state.segments = None

st.title("Customer Churn Prediction and Lifetime Value Estimation for Telecom Business Optimization")
st.markdown("""
This dashboard is part of the BDM Capstone Project by Nagarajan Venugopal (Roll: 24f1000771), IIT Madras.
It uses machine learning to predict customer churn and estimate Customer Lifetime Value (CLV) using a public Kaggle dataset.
Upload your customer data to get predictions, visualizations, and customer segments to help retain high-value customers and optimize business decisions.
""")

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Predictions & Segments", "Visualizations", "Business Insights"])

if page == "Upload Data":
    st.header("Upload Customer Data")
    st.write("Upload a CSV file with customer data (e.g., similar to the IBM Telco Churn Dataset from Kaggle). The app will clean, engineer features, and prepare it for predictions.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        df = st.session_state.df
        
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        
        if 'Churn' in df.columns:
            df["Churn"] = df["Churn"].replace({"Yes": 1, "No": 0, "None": 0})
            df["Churn"] = pd.to_numeric(df["Churn"], errors='coerce').fillna(0).astype(int)
        
        if 'ChargesPerTenure' not in df.columns:
            df['ChargesPerTenure'] = df['MonthlyCharges'] / (df['tenure'] + 1)
        
        if 'TenureGroup' not in df.columns:
            df['TenureGroup'] = pd.cut(df['tenure'], bins=[0, 12, 24, 36, 48, 60, np.inf], labels=['0-1yr', '1-2yr', '2-3yr', '3-4yr', '4-5yr', '5+yr'])
        
        if 'ServiceCount' not in df.columns:
            df['ServiceCount'] = (df[['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']] == 'Yes').sum(axis=1)
        
        if 'ARPU' not in df.columns:
            df['ARPU'] = df['TotalCharges'] / (df['tenure'] + 1)
        
        if 'RevenuePotential' not in df.columns:
            df['RevenuePotential'] = df['MonthlyCharges'] * (72 - df['tenure'])
        
        if 'ContractLength' not in df.columns:
            df['ContractLength'] = df['Contract'].map({'Month-to-month': 1, 'One year': 12, 'Two year': 24})
        
        if 'AutoPay' not in df.columns:
            df['AutoPay'] = df['PaymentMethod'].str.contains('automatic').astype(int)
        
        if 'SeniorFlag' not in df.columns:
            df['SeniorFlag'] = df['SeniorCitizen']
        
        if 'AvgChargesPerService' not in df.columns:
            df['AvgChargesPerService'] = df['MonthlyCharges'] / (df['ServiceCount'] + 1)
        
        if 'ServicesCount' not in df.columns:
            df['ServicesCount'] = df['ServiceCount']
        
        if 'RetentionScore' not in df.columns:
            df['RetentionScore'] = df['tenure'] / 72
            st.warning("RetentionScore not found in data. Computed as tenure/72 as a placeholder. Adjust based on your model.")
        
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        st.subheader("Data Info")
        buffer = StringIO()
        df.info(buf=buffer)
        st.text(buffer.getvalue())
        
        st.subheader("Shape")
        st.write(df.shape)
        
        st.subheader("Missing Values")
        st.write(df.isnull().sum())
        
        if 'Churn' in df.columns:
            st.subheader("Churn Distribution")
            st.write(df["Churn"].value_counts(normalize=True))
        
        st.session_state.processed_df = df
        st.success("Data uploaded, cleaned, and features engineered successfully! Proceed to other pages for predictions and insights.")

elif page == "Predictions & Segments":
    st.header("Customer Predictions and Segmentation")
    if st.session_state.processed_df is None:
        st.warning("Please upload data first in the 'Upload Data' page.")
    else:
        df = st.session_state.processed_df
        
        all_features = numerical_features + categorical_features
        missing_cols = set(all_features) - set(df.columns)
        if missing_cols:
            st.error(f"Missing columns in data: {missing_cols}. Please ensure all features are present or adjust feature engineering.")
        else:
            X = df[all_features]
            
            churn_preds = churn_model.predict(X)
            clv_preds = clv_model.predict(X)
            
            original_clv_preds = np.round(np.exp(clv_preds), 2)
            
            pred_df = df.copy()
            pred_df['Predicted_Churn'] = churn_preds
            pred_df['Predicted_CLV'] = original_clv_preds
            
            st.session_state.predictions = pred_df
            
            st.subheader("Prediction Results (Sample)")
            display_columns = ['customerID', 'Churn', 'Predicted_Churn', 'Predicted_CLV', 'CLV']
            if 'Churn' not in pred_df.columns:
                display_columns.remove('Churn')
            if 'CLV' not in pred_df.columns:
                display_columns.remove('CLV')
            st.dataframe(pred_df[display_columns].head(10))
            
            csv = pred_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Full Predictions", csv, "predictions.csv", "text/csv")
            
            st.subheader("Customer Segmentation")
            st.write("Customers are segmented based on predicted churn risk and CLV value. High Value: CLV > median; High Risk: Predicted Churn = 1.")
            
            median_clv = pred_df['Predicted_CLV'].median()
            pred_df['Risk_Level'] = pred_df['Predicted_Churn'].apply(lambda x: 'High Risk' if x == 1 else 'Low Risk')
            pred_df['Value_Level'] = pred_df['Predicted_CLV'].apply(lambda x: 'High Value' if x > median_clv else 'Low Value')
            pred_df['Segment'] = pred_df['Risk_Level'] + ' - ' + pred_df['Value_Level']
            
            segment_counts = pred_df['Segment'].value_counts()
            st.bar_chart(segment_counts)
            
            st.write("Segment Breakdown:")
            st.dataframe(segment_counts.reset_index().rename(columns={'index': 'Segment', 'Segment': 'Count'}))
            
            selected_segment = st.selectbox("View Customers in Segment", pred_df['Segment'].unique())
            st.dataframe(pred_df[pred_df['Segment'] == selected_segment][['customerID', 'Predicted_Churn', 'Predicted_CLV', 'Segment']])
            
            st.session_state.segments = pred_df
            st.info("Use these segments to prioritize retention efforts on 'High Risk - High Value' customers.")

elif page == "Visualizations":
    st.header("Data and Prediction Visualizations")
    if st.session_state.processed_df is None:
        st.warning("Please upload data first in the 'Upload Data' page.")
    else:
        df = st.session_state.processed_df
        
        if 'Churn' in df.columns:
            st.subheader("Actual Churn Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='Churn', data=df, ax=ax)
            ax.set_title('Actual Churn Distribution')
            st.pyplot(fig)
        
        if st.session_state.predictions is not None:
            pred_df = st.session_state.predictions
            
            st.subheader("Predicted Churn Distribution")
            fig, ax = plt.subplots()
            sns.countplot(x='Predicted_Churn', data=pred_df, ax=ax)
            ax.set_title('Predicted Churn Distribution')
            st.pyplot(fig)
            
            if 'Churn' in pred_df.columns:
                st.subheader("Confusion Matrix - Churn Prediction")
                cm = confusion_matrix(pred_df['Churn'], pred_df['Predicted_Churn'])
                fig, ax = plt.subplots()
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Churn", "Churn"])
                disp.plot(cmap="Blues", values_format="d", ax=ax)
                st.pyplot(fig)
            
            st.subheader("Predicted CLV Distribution")
            fig, ax = plt.subplots()
            sns.histplot(pred_df['Predicted_CLV'], kde=True, ax=ax)
            ax.set_title('Predicted CLV Distribution')
            st.pyplot(fig)
            
            st.subheader("Predicted CLV by Tenure Group")
            fig, ax = plt.subplots()
            sns.boxplot(x='TenureGroup', y='Predicted_CLV', data=pred_df, ax=ax)
            ax.set_title('Predicted CLV by Tenure Group')
            st.pyplot(fig)

elif page == "Business Insights":
    st.header("Business Insights and Recommendations")
    if st.session_state.predictions is None:
        st.warning("Please run predictions first in the 'Predictions & Segments' page.")
    else:
        pred_df = st.session_state.predictions
        
        if 'Churn' in pred_df.columns and pred_df['Churn'].notna().all():
            st.subheader("Estimated Business Impact")
            y_true = pred_df['Churn']
            y_pred = pred_df['Predicted_Churn']
            clv = pred_df['Predicted_CLV']
            
            risk_df = pd.DataFrame({
                "TrueChurn": y_true,
                "PredChurn": y_pred,
                "CLV": clv
            })
            
            fn_loss = risk_df[(risk_df["TrueChurn"] == 1) & (risk_df["PredChurn"] == 0)]["CLV"].sum()
            fp_cost = risk_df[(risk_df["TrueChurn"] == 0) & (risk_df["PredChurn"] == 1)]["CLV"].sum() * 0.05
            tp_savings = risk_df[(risk_df["TrueChurn"] == 1) & (risk_df["PredChurn"] == 1)]["CLV"].sum() * 0.3
            
            st.write(f"Lost Revenue from False Negatives (Missed Churn): ₹{fn_loss:,.0f}")
            st.write(f"Retention Cost for False Positives: ₹{fp_cost:,.0f}")
            st.write(f"Potential Savings from True Positives (Retained Churn): ₹{tp_savings:,.0f}")
            
            net_impact = tp_savings - fp_cost - fn_loss
            st.write(f"Net Business Impact: ₹{net_impact:,.0f}")
        else:
            st.info("Actual Churn labels not available in uploaded data. Business impact calculation skipped.")
        
        st.subheader("Recommendations")
        st.markdown("""
        - **Focus on High Risk - High Value Customers**: Offer personalized discounts or upgrades to reduce churn.
        - **Monitor Low Value Segments**: Consider cost-effective strategies like automated emails.
        - **Use Insights for Optimization**: Target marketing efforts on high-value customers to increase retention and profits.
        - This aligns with the project's goal to help Telco Communications Ltd. save on acquisition costs and prioritize valuable customers.
        """)