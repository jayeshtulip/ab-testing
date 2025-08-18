import streamlit as st
import requests
import json
import time
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Loan Default Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_URL = "http://localhost:8001"

def check_api_health():
    '''Check if API is running'''
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except:
        return False, None

def make_prediction(features):
    '''Make prediction using the API'''
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json={"features": features},
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}

def main():
    # Header
    st.title("🏦 Loan Default Prediction System")
    st.markdown("### AI-powered loan default risk assessment")
    
    # Sidebar for API status
    with st.sidebar:
        st.header(" System Status")
        
        # Check API health
        api_healthy, health_data = check_api_health()
        
        if api_healthy:
            st.success("✅ API Connected")
            if health_data:
                st.json(health_data)
        else:
            st.error(" API Not Available")
            st.warning("Please start the API server on port 8001")
            
        st.markdown("---")
        st.markdown("**📖 Instructions:**")
        st.markdown("1. Fill in the loan application details")
        st.markdown("2. Click 'Predict Risk' to get assessment")
        st.markdown("3. Review the prediction and confidence score")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header(" Loan Application Form")
        
        # Create form
        with st.form("loan_form"):
            st.subheader("Applicant Information")
            
            # Key features
            col_a, col_b = st.columns(2)
            
            with col_a:
                attribute1 = st.selectbox(
                    "Checking Account Status",
                    ["A11", "A12", "A13", "A14"],
                    help="A11: <0 DM, A12: 0-200 DM, A13: >200 DM, A14: No account"
                )
                
                duration = st.number_input(
                    "Loan Duration (months)",
                    min_value=1, max_value=72, value=24,
                    help="Duration of the loan in months"
                )
                
                attribute3 = st.selectbox(
                    "Credit History",
                    ["A30", "A31", "A32", "A33", "A34"],
                    index=2,
                    help="A30: No credits, A31: All paid, A32: Existing paid, A33: Delay, A34: Critical"
                )
                
            with col_b:
                credit_amount = st.number_input(
                    "Credit Amount (DM)",
                    min_value=100, max_value=20000, value=3500,
                    help="Amount of credit requested"
                )
                
                age = st.number_input(
                    "Age (years)",
                    min_value=18, max_value=80, value=35,
                    help="Age of the applicant"
                )
                
                installment_rate = st.slider(
                    "Installment Rate (%)",
                    min_value=1, max_value=10, value=4,
                    help="Installment rate in percentage of disposable income"
                )
            
            # Additional features (optional)
            with st.expander(" Additional Information (Optional)", expanded=False):
                col_c, col_d = st.columns(2)
                
                with col_c:
                    purpose = st.selectbox("Purpose", ["A43", "A40", "A41", "A42", "A44", "A45", "A46"])
                    savings = st.selectbox("Savings Account", ["A61", "A62", "A63", "A64", "A65"])
                    employment = st.selectbox("Employment", ["A71", "A72", "A73", "A74", "A75"])
                    
                with col_d:
                    personal_status = st.selectbox("Personal Status", ["A91", "A92", "A93", "A94", "A95"])
                    housing = st.selectbox("Housing", ["A151", "A152", "A153"])
                    job = st.selectbox("Job", ["A171", "A172", "A173", "A174"])
            
            # Submit button
            submitted = st.form_submit_button(" Predict Default Risk", use_container_width=True)
            
            if submitted:
                if not api_healthy:
                    st.error(" Cannot make prediction - API not available")
                else:
                    # Prepare features
                    features = {
                        "Attribute1": attribute1,
                        "Attribute2": duration,
                        "Attribute3": attribute3,
                        "Attribute5": credit_amount,
                        "Attribute13": age,
                        "Attribute8": installment_rate
                    }
                    
                    # Show loading spinner
                    with st.spinner(" Analyzing loan application..."):
                        time.sleep(1)  # Simulate processing time
                        success, result = make_prediction(features)
                    
                    # Display results in the right column
                    with col2:
                        st.header(" Prediction Results")
                        
                        if success:
                            prediction = result.get("prediction", "unknown")
                            probability = result.get("probability", 0)
                            
                            # Risk assessment
                            if prediction == "good":
                                st.success(" LOW RISK")
                                st.balloons()
                                risk_color = "green"
                                risk_message = "This loan application shows low default risk."
                            else:
                                st.error(" HIGH RISK")
                                risk_color = "red"
                                risk_message = "This loan application shows high default risk."
                            
                            # Confidence metrics
                            st.metric(
                                label="Confidence Score",
                                value=f"{probability:.1%}",
                                delta=f"Model certainty"
                            )
                            
                            # Risk gauge
                            st.markdown(f"**Risk Level:** :{risk_color}[{prediction.upper()}]")
                            st.progress(probability)
                            
                            # Detailed results
                            st.markdown("---")
                            st.markdown("** Detailed Analysis:**")
                            st.write(risk_message)
                            
                            # Technical details
                            with st.expander(" Technical Details"):
                                st.json({
                                    "prediction": prediction,
                                    "probability": f"{probability:.4f}",
                                    "model_version": result.get("model_version", "unknown"),
                                    "timestamp": datetime.fromtimestamp(result.get("timestamp", time.time())).strftime("%Y-%m-%d %H:%M:%S"),
                                    "input_features": features
                                })
                        else:
                            st.error(" Prediction Failed")
                            st.write("Error details:", result)
    
    # Instructions and information
    with col2:
        if 'submitted' not in locals() or not submitted:
            st.header(" How to Use")
            st.markdown("""
            **Step 1:** Fill out the loan application form with applicant details.
            
            **Step 2:** Click 'Predict Default Risk' to get an AI assessment.
            
            **Step 3:** Review the risk prediction and confidence score.
            
            **Understanding Results:**
            -  **LOW RISK**: Likely to repay the loan
            -  **HIGH RISK**: Higher chance of default
            
            **Confidence Score:** How certain the AI model is about its prediction.
            """)
            
            st.markdown("---")
            st.info(" **Tip:** The more accurate information you provide, the better the prediction accuracy.")

if __name__ == "__main__":
    main()
