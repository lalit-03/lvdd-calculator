import streamlit as st
import pandas as pd
import numpy as np
import math
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from streamlit_gsheets import GSheetsConnection

# Page configuration
st.set_page_config(
    page_title="LVDD Probability Calculator",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model coefficients from the Excel calculator
MODEL_COEFFICIENTS = {
    'Age (Years)': 0.1025470193566646,
    'BMI (Kg/m²)': -0.05226483119313179,
    'HDL (mg/dL)': 0.00680820478879949,
    'LDL (mg/dL)': -0.006399508517213281,
    'Total Cholesterol (mg/dL)': 0.006442423537169774,
    'HbA1c (%)': -0.07148207110249861,
    'TGI': 1.695718833984679
}

# Model intercept
INTERCEPT = -19.22888123215025

# Clinical cutoffs
SCREENING_CUTOFF = 0.1  # Rule-out cutoff (high sensitivity & NPV)
RULE_IN_CUTOFF = 0.8   # Rule-in cutoff (high specificity & PPV)

def calculate_lvdd_probability(age, bmi, hdl, ldl, total_chol, hba1c, tgi):
    """
    Calculate LVDD probability using logistic regression model
    """
    # Calculate linear predictor (log-odds)
    linear_predictor = (
        INTERCEPT +
        MODEL_COEFFICIENTS['Age (Years)'] * age +
        MODEL_COEFFICIENTS['BMI (Kg/m²)'] * bmi +
        MODEL_COEFFICIENTS['HDL (mg/dL)'] * hdl +
        MODEL_COEFFICIENTS['LDL (mg/dL)'] * ldl +
        MODEL_COEFFICIENTS['Total Cholesterol (mg/dL)'] * total_chol +
        MODEL_COEFFICIENTS['HbA1c (%)'] * hba1c +
        MODEL_COEFFICIENTS['TGI'] * tgi
    )
    
    # Convert to probability using logistic function
    probability = 1 / (1 + math.exp(-linear_predictor))
    
    return probability, linear_predictor

def get_risk_classification(probability):
    """
    Classify risk based on probability thresholds
    """
    if probability < SCREENING_CUTOFF:
        return "Low Risk", "green", "Rule-out: Low probability of LVDD"
    elif probability > RULE_IN_CUTOFF:
        return "High Risk", "red", "Rule-in: High probability of LVDD"
    else:
        return "Intermediate Risk", "orange", "Requires further clinical assessment"

def create_probability_gauge(probability):
    """
    Create a gauge chart for probability visualization
    """
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = probability * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "LVDD Probability (%)"},
        delta = {'reference': 50},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, SCREENING_CUTOFF*100], 'color': "lightgreen"},
                {'range': [SCREENING_CUTOFF*100, RULE_IN_CUTOFF*100], 'color': "lightyellow"},
                {'range': [RULE_IN_CUTOFF*100, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': RULE_IN_CUTOFF*100
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def save_to_google_sheets(data):
    """
    Save calculation results to Google Sheets using Streamlit data connector
    """
    try:
        # Create connection to Google Sheets
        conn = st.connection("gsheets", type=GSheetsConnection)

        # Convert new row to DataFrame
        new_df = pd.DataFrame([data])

        # Read existing sheet
        existing_data = conn.read(worksheet="LVDD_Calculations")

        # If existing_data is None or empty, just use new_df
        if existing_data is None or existing_data.empty:
            updated_df = new_df
        else:
            # Append new row
            updated_df = pd.concat([existing_data, new_df], ignore_index=True)

        # Write full dataframe back
        conn.update(worksheet="LVDD_Calculations", data=updated_df)

        return True
    except Exception as e:
        st.error(f"Error saving to Google Sheets: {str(e)}")
        return False


# Main app
def main():
    # Title and description
    st.title("🫀 LVDD Probability Calculator")
    st.markdown("### Left Ventricular Diastolic Dysfunction Risk Assessment")
    st.markdown("This calculator uses a validated continuous model to predict the probability of LVDD based on clinical parameters.")
    
    # Sidebar for instructions
    with st.sidebar:
        st.header("Instructions")
        st.markdown("""
        **How to use:**
        1. Enter patient values in the form
        2. View the calculated probability
        3. Interpret results using cutoffs:
           - < 10%: Rule-out (Low risk)
           - > 80%: Rule-in (High risk)
           - 10-80%: Intermediate risk
        4. Results are automatically saved
        
        **Clinical Cutoffs:**
        - **Screening (10%)**: High sensitivity & NPV for rule-out
        - **Rule-in (80%)**: High specificity & PPV for diagnosis
        """)
        
        st.header("Model Information")
        st.markdown("**Variables included:**")
        for var in MODEL_COEFFICIENTS.keys():
            st.markdown(f"• {var}")
    
    # Create two columns for input and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("Patient Information")
        
        # Patient demographics
        with st.form("patient_form"):
            st.subheader("Demographics & Clinical Values")
            
            # Patient ID and timestamp
            patient_id = st.text_input("Patient ID (optional)", value="")
            
            # Clinical parameters
            age = st.number_input("Age (Years)", min_value=0, max_value=120, value=65, step=1)
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            
            col1a, col1b = st.columns(2)
            with col1a:
                hdl = st.number_input("HDL (mg/dL)", min_value=0, max_value=200, value=50, step=1)
                ldl = st.number_input("LDL (mg/dL)", min_value=0, max_value=400, value=100, step=1)
            with col1b:
                total_chol = st.number_input("Total Cholesterol (mg/dL)", min_value=0, max_value=500, value=200, step=1)
                hba1c = st.number_input("HbA1c (%)", min_value=3.0, max_value=15.0, value=5.7, step=0.1)
            
            tgi = st.number_input("TGI", min_value=0.0, max_value=10.0, value=1.0, step=0.1, 
                                help="Triglyceride-Glucose Index")
            
            # Form submission
            calculate_button = st.form_submit_button("Calculate LVDD Probability", type="primary")
    
    with col2:
        st.header("Results")
        
        if calculate_button:
            # Calculate probability
            probability, linear_predictor = calculate_lvdd_probability(
                age, bmi, hdl, ldl, total_chol, hba1c, tgi
            )
            
            # Get risk classification
            risk_class, color, interpretation = get_risk_classification(probability)
            
            # Display results
            st.subheader("Probability Results")
            
            # Probability display
            st.metric(
                label="LVDD Probability", 
                value=f"{probability:.1%}",
                delta=f"Linear Predictor: {linear_predictor:.4f}"
            )
            
            # Risk classification
            st.markdown(f"**Risk Classification:** :{color}[{risk_class}]")
            st.info(interpretation)
            
            # Gauge chart
            st.plotly_chart(create_probability_gauge(probability), use_container_width=True)
            
            # Clinical interpretation
            st.subheader("Clinical Interpretation")
            if probability < SCREENING_CUTOFF:
                st.success("✅ **Low probability of LVDD** - Consider ruling out LVDD")
            elif probability > RULE_IN_CUTOFF:
                st.error("⚠️ **High probability of LVDD** - Consider further cardiac evaluation")
            else:
                st.warning("🔍 **Intermediate risk** - Clinical judgment and additional testing recommended")
            
            # Prepare data for saving
            calculation_data = {
                'timestamp': datetime.now().isoformat(),
                'patient_id': patient_id if patient_id else f"Patient_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'age': age,
                'bmi': bmi,
                'hdl': hdl,
                'ldl': ldl,
                'total_cholesterol': total_chol,
                'hba1c': hba1c,
                'tgi': tgi,
                'linear_predictor': linear_predictor,
                'probability': probability,
                'risk_classification': risk_class,
                'interpretation': interpretation
            }
            
            # Save to Google Sheets
            with st.spinner("Saving results..."):
                if save_to_google_sheets(calculation_data):
                    st.success("✅ Results saved successfully!")
                else:
                    st.warning("⚠️ Could not save to Google Sheets. Results calculated successfully.")
                    
                    # Offer CSV download as backup
                    csv_data = pd.DataFrame([calculation_data]).to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name=f"lvdd_calculation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
    
    # Additional features
    st.markdown("---")
    
    # Historical data view
    if st.checkbox("📊 View Historical Calculations"):
        try:
            conn = st.connection("gsheets", type=GSheetsConnection)
            historical_data = conn.read(worksheet="LVDD_Calculations")
            
            if not historical_data.empty:
                st.subheader("Historical Calculations")
                
                # Display recent calculations
                st.dataframe(historical_data.tail(10), use_container_width=True)
                
                # Summary statistics
                col3, col4, col5 = st.columns(3)
                with col3:
                    st.metric("Total Calculations", len(historical_data))
                with col4:
                    avg_prob = historical_data['probability'].mean()
                    st.metric("Average Probability", f"{avg_prob:.1%}")
                with col5:
                    high_risk_count = len(historical_data[historical_data['probability'] > RULE_IN_CUTOFF])
                    st.metric("High Risk Cases", f"{high_risk_count}")
                
                # Probability distribution
                if len(historical_data) > 1:
                    fig_hist = px.histogram(
                        historical_data, 
                        x='probability',
                        title="Distribution of LVDD Probabilities",
                        nbins=20
                    )
                    fig_hist.add_vline(x=SCREENING_CUTOFF, line_dash="dash", line_color="green", 
                                      annotation_text="Screening Cutoff")
                    fig_hist.add_vline(x=RULE_IN_CUTOFF, line_dash="dash", line_color="red", 
                                      annotation_text="Rule-in Cutoff")
                    st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No historical data available yet.")
                
        except Exception as e:
            st.error(f"Error loading historical data: {str(e)}")
    
    # Model details
    if st.checkbox("🔬 Show Model Details"):
        st.subheader("Model Coefficients")
        coeff_df = pd.DataFrame([
            {"Variable": var, "Beta Coefficient": coeff, "Interpretation": 
             "Higher values increase LVDD probability" if coeff > 0 else "Higher values decrease LVDD probability"}
            for var, coeff in MODEL_COEFFICIENTS.items()
        ])
        st.dataframe(coeff_df, use_container_width=True)
        
        st.subheader("Model Formula")
        st.latex(r'''
        \text{Linear Predictor} = \beta_0 + \sum_{i=1}^{n} \beta_i \times X_i
        ''')
        st.latex(r'''
        \text{Probability} = \frac{1}{1 + e^{-\text{Linear Predictor}}}
        ''')

if __name__ == "__main__":
    main()