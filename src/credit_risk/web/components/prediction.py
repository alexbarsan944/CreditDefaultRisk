import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import logging
import plotly.express as px
import plotly.graph_objects as go
import json
from typing import Dict, List, Any

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add the parent directory to the path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

# Import the necessary modules
from credit_risk.web.model_utils import (
    get_application_prediction,
    load_model_and_selector
)
from credit_risk.web.app import (
    set_page_config,
    setup_sidebar
)

# Set page config
set_page_config("Prediction")

# Sample application schema to guide input
SAMPLE_APPLICATION = {
    "categorical_features": {
        "NAME_CONTRACT_TYPE": ["Cash loans", "Revolving loans"],
        "CODE_GENDER": ["F", "M"],
        "FLAG_OWN_CAR": ["Y", "N"],
        "FLAG_OWN_REALTY": ["Y", "N"],
        "NAME_INCOME_TYPE": ["Working", "State servant", "Commercial associate", "Pensioner", "Unemployed"],
        "NAME_EDUCATION_TYPE": ["Higher education", "Secondary / secondary special", "Incomplete higher", "Lower secondary", "Academic degree"],
        "NAME_FAMILY_STATUS": ["Married", "Single / not married", "Civil marriage", "Separated", "Widow"],
        "NAME_HOUSING_TYPE": ["House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment", "Co-op apartment"],
        "OCCUPATION_TYPE": ["Laborers", "Core staff", "Sales staff", "Managers", "Drivers", "High skill tech staff", "Accountants", "Medicine staff", "Cooking staff", "Security staff", "Cleaning staff", "Private service staff", "Low-skill Laborers", "Waiters/barmen staff", "Secretaries", "HR staff", "Realty agents"]
    },
    "numerical_features": {
        "AMT_INCOME_TOTAL": {"min": 25000, "max": 1000000, "default": 150000, "step": 5000},
        "AMT_CREDIT": {"min": 50000, "max": 2000000, "default": 500000, "step": 10000},
        "AMT_ANNUITY": {"min": 5000, "max": 150000, "default": 25000, "step": 1000},
        "AMT_GOODS_PRICE": {"min": 40000, "max": 1800000, "default": 450000, "step": 10000},
        "DAYS_BIRTH": {"min": -25000, "max": -8000, "default": -16000, "step": 100, "help": "Days before current day (negative). Default -16000 is about 43 years old."},
        "DAYS_EMPLOYED": {"min": -18000, "max": 0, "default": -2000, "step": 100, "help": "Days before current day (negative). Default -2000 is about 5.5 years of employment."},
        "CNT_CHILDREN": {"min": 0, "max": 10, "default": 0, "step": 1},
        "CNT_FAM_MEMBERS": {"min": 1, "max": 15, "default": 2, "step": 1}
    }
}

def create_application_form() -> Dict[str, Any]:
    """
    Create a form for entering application data.
    
    Returns
    -------
    Dict[str, Any]
        Dictionary of application data
    """
    application_data = {}
    
    # Create tabs for different categories of features
    tab1, tab2 = st.tabs(["Personal Information", "Financial Information"])
    
    with tab1:
        st.subheader("Personal Information")
        
        # Categorical features
        col1, col2 = st.columns(2)
        
        with col1:
            application_data["CODE_GENDER"] = st.selectbox(
                "Gender",
                options=SAMPLE_APPLICATION["categorical_features"]["CODE_GENDER"],
                index=0
            )
            
            application_data["FLAG_OWN_CAR"] = st.selectbox(
                "Owns a Car",
                options=SAMPLE_APPLICATION["categorical_features"]["FLAG_OWN_CAR"],
                index=1
            )
            
            application_data["FLAG_OWN_REALTY"] = st.selectbox(
                "Owns Real Estate",
                options=SAMPLE_APPLICATION["categorical_features"]["FLAG_OWN_REALTY"],
                index=0
            )
            
            application_data["NAME_INCOME_TYPE"] = st.selectbox(
                "Income Type",
                options=SAMPLE_APPLICATION["categorical_features"]["NAME_INCOME_TYPE"],
                index=0
            )
        
        with col2:
            application_data["NAME_EDUCATION_TYPE"] = st.selectbox(
                "Education Level",
                options=SAMPLE_APPLICATION["categorical_features"]["NAME_EDUCATION_TYPE"],
                index=0
            )
            
            application_data["NAME_FAMILY_STATUS"] = st.selectbox(
                "Family Status",
                options=SAMPLE_APPLICATION["categorical_features"]["NAME_FAMILY_STATUS"],
                index=0
            )
            
            application_data["NAME_HOUSING_TYPE"] = st.selectbox(
                "Housing Type",
                options=SAMPLE_APPLICATION["categorical_features"]["NAME_HOUSING_TYPE"],
                index=0
            )
            
            application_data["OCCUPATION_TYPE"] = st.selectbox(
                "Occupation",
                options=SAMPLE_APPLICATION["categorical_features"]["OCCUPATION_TYPE"],
                index=0
            )
        
        # Numerical features
        col1, col2 = st.columns(2)
        
        with col1:
            application_data["CNT_CHILDREN"] = st.number_input(
                "Number of Children",
                min_value=SAMPLE_APPLICATION["numerical_features"]["CNT_CHILDREN"]["min"],
                max_value=SAMPLE_APPLICATION["numerical_features"]["CNT_CHILDREN"]["max"],
                value=SAMPLE_APPLICATION["numerical_features"]["CNT_CHILDREN"]["default"],
                step=SAMPLE_APPLICATION["numerical_features"]["CNT_CHILDREN"]["step"]
            )
            
            application_data["CNT_FAM_MEMBERS"] = st.number_input(
                "Number of Family Members",
                min_value=SAMPLE_APPLICATION["numerical_features"]["CNT_FAM_MEMBERS"]["min"],
                max_value=SAMPLE_APPLICATION["numerical_features"]["CNT_FAM_MEMBERS"]["max"],
                value=SAMPLE_APPLICATION["numerical_features"]["CNT_FAM_MEMBERS"]["default"],
                step=SAMPLE_APPLICATION["numerical_features"]["CNT_FAM_MEMBERS"]["step"]
            )
        
        with col2:
            days_birth = st.slider(
                "Age (years)",
                min_value=20,
                max_value=70,
                value=43,
                step=1
            )
            application_data["DAYS_BIRTH"] = -days_birth * 365
            
            years_employed = st.slider(
                "Employment Duration (years)",
                min_value=0,
                max_value=50,
                value=5,
                step=1
            )
            application_data["DAYS_EMPLOYED"] = -years_employed * 365 if years_employed > 0 else 0
    
    with tab2:
        st.subheader("Financial Information")
        
        # Loan details
        col1, col2 = st.columns(2)
        
        with col1:
            application_data["NAME_CONTRACT_TYPE"] = st.selectbox(
                "Contract Type",
                options=SAMPLE_APPLICATION["categorical_features"]["NAME_CONTRACT_TYPE"],
                index=0
            )
            
            application_data["AMT_INCOME_TOTAL"] = st.number_input(
                "Total Income",
                min_value=SAMPLE_APPLICATION["numerical_features"]["AMT_INCOME_TOTAL"]["min"],
                max_value=SAMPLE_APPLICATION["numerical_features"]["AMT_INCOME_TOTAL"]["max"],
                value=SAMPLE_APPLICATION["numerical_features"]["AMT_INCOME_TOTAL"]["default"],
                step=SAMPLE_APPLICATION["numerical_features"]["AMT_INCOME_TOTAL"]["step"]
            )
            
            application_data["AMT_CREDIT"] = st.number_input(
                "Credit Amount",
                min_value=SAMPLE_APPLICATION["numerical_features"]["AMT_CREDIT"]["min"],
                max_value=SAMPLE_APPLICATION["numerical_features"]["AMT_CREDIT"]["max"],
                value=SAMPLE_APPLICATION["numerical_features"]["AMT_CREDIT"]["default"],
                step=SAMPLE_APPLICATION["numerical_features"]["AMT_CREDIT"]["step"]
            )
        
        with col2:
            application_data["AMT_ANNUITY"] = st.number_input(
                "Annuity Amount",
                min_value=SAMPLE_APPLICATION["numerical_features"]["AMT_ANNUITY"]["min"],
                max_value=SAMPLE_APPLICATION["numerical_features"]["AMT_ANNUITY"]["max"],
                value=SAMPLE_APPLICATION["numerical_features"]["AMT_ANNUITY"]["default"],
                step=SAMPLE_APPLICATION["numerical_features"]["AMT_ANNUITY"]["step"]
            )
            
            application_data["AMT_GOODS_PRICE"] = st.number_input(
                "Goods Price",
                min_value=SAMPLE_APPLICATION["numerical_features"]["AMT_GOODS_PRICE"]["min"],
                max_value=SAMPLE_APPLICATION["numerical_features"]["AMT_GOODS_PRICE"]["max"],
                value=SAMPLE_APPLICATION["numerical_features"]["AMT_GOODS_PRICE"]["default"],
                step=SAMPLE_APPLICATION["numerical_features"]["AMT_GOODS_PRICE"]["step"]
            )
    
    # Add some derived features that might be expected by the model
    application_data["DAYS_EMPLOYED_PERC"] = application_data["DAYS_EMPLOYED"] / application_data["DAYS_BIRTH"]
    application_data["INCOME_CREDIT_PERC"] = application_data["AMT_INCOME_TOTAL"] / application_data["AMT_CREDIT"]
    application_data["INCOME_PER_PERSON"] = application_data["AMT_INCOME_TOTAL"] / application_data["CNT_FAM_MEMBERS"]
    application_data["ANNUITY_INCOME_PERC"] = application_data["AMT_ANNUITY"] / application_data["AMT_INCOME_TOTAL"]
    application_data["PAYMENT_RATE"] = application_data["AMT_ANNUITY"] / application_data["AMT_CREDIT"]
    
    return application_data

def display_prediction_results(prediction_results: Dict[str, Any], application_data: Dict[str, Any]) -> None:
    """
    Display prediction results.
    
    Parameters
    ----------
    prediction_results : Dict[str, Any]
        Results of the prediction
    application_data : Dict[str, Any]
        Application data
    """
    # Get prediction and probability
    probability = prediction_results["probability"]
    prediction = prediction_results["prediction"]
    
    # Create columns for prediction and probability
    col1, col2 = st.columns(2)
    
    with col1:
        # Display prediction
        if prediction == 1:
            st.error("### 🔴 High Risk of Default")
        else:
            st.success("### 🟢 Low Risk of Default")
    
    with col2:
        # Display probability gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            title={"text": "Default Probability"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkred" if probability > 0.5 else "darkgreen"},
                "steps": [
                    {"range": [0, 50], "color": "lightgreen"},
                    {"range": [50, 100], "color": "lightcoral"}
                ],
                "threshold": {
                    "line": {"color": "black", "width": 4},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display top features influencing the prediction
    if "top_features" in prediction_results and prediction_results["top_features"] is not None:
        st.subheader("Top Features Influencing Prediction")
        
        top_features = prediction_results["top_features"]
        
        # Create bar chart of feature importance
        fig = px.bar(
            top_features,
            y="feature",
            x="importance",
            orientation="h",
            title="Feature Importance",
            color="importance",
            color_continuous_scale=["green", "yellow", "red"]
        )
        
        fig.update_layout(
            yaxis_title="",
            xaxis_title="Importance",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display table with feature values
        feature_values = []
        for feature_name in top_features["feature"]:
            if feature_name in application_data:
                feature_values.append({
                    "Feature": feature_name,
                    "Value": application_data[feature_name]
                })
        
        if feature_values:
            st.dataframe(pd.DataFrame(feature_values))
    
def main():
    """Main function for the Prediction page."""
    
    # Set up sidebar
    backend, raw_data_path, current_step = setup_sidebar(current_page="Prediction")
    
    # Display page header
    st.title("Credit Default Risk - Prediction")
    st.write(
        """
        This page allows you to make predictions on new loan applications using the trained model.
        You can either use a model from the previous step or load a saved model.
        """
    )
    
    # Model selection
    st.header("Model Selection")
    
    model_source = st.radio(
        "Select model source",
        options=["Use model from previous step", "Load saved model"],
        index=0
    )
    
    model = None
    feature_selector = None
    
    if model_source == "Use model from previous step":
        if "model_results" in st.session_state:
            model = st.session_state.model_results["model"]
            feature_selector = st.session_state.model_results.get("feature_selector")
            model_type = st.session_state.model_results["model_type"]
            
            st.success(f"Using {model_type.upper()} model from previous step")
        else:
            st.warning("No model found from previous step. Please train a model first or load a saved model.")
    else:
        # Load saved model
        model_path = st.text_input(
            "Model path",
            value="models/lightgbm_model.joblib"
        )
        
        selector_path = st.text_input(
            "Feature selector path (optional)",
            value="models/lightgbm_feature_selector.joblib"
        )
        
        if st.button("Load Model"):
            try:
                if os.path.exists(model_path):
                    if os.path.exists(selector_path):
                        model, feature_selector = load_model_and_selector(model_path, selector_path)
                        st.success("Model and feature selector loaded successfully!")
                    else:
                        from credit_risk.models import CreditRiskModel
                        model = CreditRiskModel.load(model_path)
                        st.success("Model loaded successfully! (No feature selector found)")
                else:
                    st.error(f"Model file not found: {model_path}")
            except Exception as e:
                st.error(f"Error loading model: {str(e)}")
                logger.error(f"Model loading error: {str(e)}", exc_info=True)
    
    # Application form
    if model is not None:
        st.header("Application Form")
        st.write("Enter the details of the loan application:")
        
        # Load presets
        load_preset = st.checkbox("Load preset application", value=False)
        
        if load_preset:
            preset_type = st.selectbox(
                "Select preset",
                options=["Low Risk Applicant", "Medium Risk Applicant", "High Risk Applicant"]
            )
            
            # Define presets
            presets = {
                "Low Risk Applicant": {
                    "CODE_GENDER": "F",
                    "FLAG_OWN_CAR": "Y",
                    "FLAG_OWN_REALTY": "Y",
                    "NAME_INCOME_TYPE": "Working",
                    "NAME_EDUCATION_TYPE": "Higher education",
                    "NAME_FAMILY_STATUS": "Married",
                    "NAME_HOUSING_TYPE": "House / apartment",
                    "OCCUPATION_TYPE": "High skill tech staff",
                    "CNT_CHILDREN": 0,
                    "CNT_FAM_MEMBERS": 2,
                    "DAYS_BIRTH": -16425,  # 45 years
                    "DAYS_EMPLOYED": -7300,  # 20 years
                    "NAME_CONTRACT_TYPE": "Cash loans",
                    "AMT_INCOME_TOTAL": 250000,
                    "AMT_CREDIT": 500000,
                    "AMT_ANNUITY": 30000,
                    "AMT_GOODS_PRICE": 450000
                },
                "Medium Risk Applicant": {
                    "CODE_GENDER": "M",
                    "FLAG_OWN_CAR": "Y",
                    "FLAG_OWN_REALTY": "N",
                    "NAME_INCOME_TYPE": "Working",
                    "NAME_EDUCATION_TYPE": "Secondary / secondary special",
                    "NAME_FAMILY_STATUS": "Married",
                    "NAME_HOUSING_TYPE": "Rented apartment",
                    "OCCUPATION_TYPE": "Laborers",
                    "CNT_CHILDREN": 2,
                    "CNT_FAM_MEMBERS": 4,
                    "DAYS_BIRTH": -12775,  # 35 years
                    "DAYS_EMPLOYED": -1825,  # 5 years
                    "NAME_CONTRACT_TYPE": "Cash loans",
                    "AMT_INCOME_TOTAL": 120000,
                    "AMT_CREDIT": 600000,
                    "AMT_ANNUITY": 40000,
                    "AMT_GOODS_PRICE": 550000
                },
                "High Risk Applicant": {
                    "CODE_GENDER": "M",
                    "FLAG_OWN_CAR": "N",
                    "FLAG_OWN_REALTY": "N",
                    "NAME_INCOME_TYPE": "Working",
                    "NAME_EDUCATION_TYPE": "Lower secondary",
                    "NAME_FAMILY_STATUS": "Single / not married",
                    "NAME_HOUSING_TYPE": "Rented apartment",
                    "OCCUPATION_TYPE": "Low-skill Laborers",
                    "CNT_CHILDREN": 0,
                    "CNT_FAM_MEMBERS": 1,
                    "DAYS_BIRTH": -9125,  # 25 years
                    "DAYS_EMPLOYED": -365,  # 1 year
                    "NAME_CONTRACT_TYPE": "Revolving loans",
                    "AMT_INCOME_TOTAL": 60000,
                    "AMT_CREDIT": 300000,
                    "AMT_ANNUITY": 30000,
                    "AMT_GOODS_PRICE": 280000
                }
            }
            
            selected_preset = presets[preset_type]
            
            # Add derived features
            selected_preset["DAYS_EMPLOYED_PERC"] = selected_preset["DAYS_EMPLOYED"] / selected_preset["DAYS_BIRTH"]
            selected_preset["INCOME_CREDIT_PERC"] = selected_preset["AMT_INCOME_TOTAL"] / selected_preset["AMT_CREDIT"]
            selected_preset["INCOME_PER_PERSON"] = selected_preset["AMT_INCOME_TOTAL"] / selected_preset["CNT_FAM_MEMBERS"]
            selected_preset["ANNUITY_INCOME_PERC"] = selected_preset["AMT_ANNUITY"] / selected_preset["AMT_INCOME_TOTAL"]
            selected_preset["PAYMENT_RATE"] = selected_preset["AMT_ANNUITY"] / selected_preset["AMT_CREDIT"]
            
            # Display preset data
            st.json(json.dumps(selected_preset, indent=2))
            
            application_data = selected_preset
        else:
            # Create application form
            application_data = create_application_form()
        
        # Make prediction button
        if st.button("Make Prediction", type="primary"):
            try:
                # Convert application data to DataFrame
                application_df = pd.DataFrame([application_data])
                
                # Make prediction
                prediction_results = get_application_prediction(
                    model,
                    application_df,
                    feature_selector
                )
                
                # Display results
                st.header("Prediction Results")
                display_prediction_results(prediction_results, application_data)
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                logger.error(f"Prediction error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()