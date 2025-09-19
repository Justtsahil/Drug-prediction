import streamlit as st
import pandas as pd
import joblib
import json
import warnings
import plotly.express as px
from streamlit_card import card
# Add these imports for PDF generation
from fpdf import FPDF
from datetime import datetime
warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Page configuration
st.set_page_config(
    page_title="Medical Outcome Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E88E5;
        font-weight: 700;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        font-weight: 600;
        border-bottom: 1px solid #f0f0f0;
        padding-bottom: 1rem;
    }
    .prediction-header {
        font-size: 1.8rem;
        color: #1E88E5;
        font-weight: 600;
        margin-top: 2rem;
    }
    .card {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 20px;
        margin-bottom: 20px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .section-divider {
        height: 2px;
        background-color: #f0f0f0;
        margin: 15px 0;
    }
</style>
""", unsafe_allow_html=True)

# Utility Functions
def load_model(model_path):
    return joblib.load(model_path)

def encode_input_features(input_features, feature_list):
    return pd.DataFrame([feature_list], columns=input_features)

def decode_predictions(predictions, encoders):
    decoded_predictions = {}
    for col in predictions.columns:
        encoder = encoders.get(col)
        if encoder:
            decoded_value = encoder.inverse_transform([predictions[col].iloc[0]])[0]
            decoded_predictions[col] = decoded_value
        else:
            decoded_predictions[col] = "Error: No encoder found"
    return decoded_predictions

def get_mapped_outputs(predicted_disease, mappings):
    return mappings.get(predicted_disease, {})

def generate_pdf_report(patient_data, prediction_results):
    """Generate a compact professional medical report PDF with patient data and prediction results"""
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.set_margins(10, 10, 10)  # Smaller margins (left, top, right)
    pdf.add_page()
    
    # Add headers and styling - more compact
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(190, 6, "Medical Report", ln=True, align='C')
    
    # Add patient name in the header if available
    patient_name = patient_data.get('Patient_Name', '')
    if patient_name:
        pdf.set_font("Arial", 'B', 10)
        pdf.cell(190, 4, f"Patient: {patient_name}", ln=True, align='C')
    
    pdf.set_font("Arial", 'I', 8)
    pdf.cell(190, 4, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.line(10, 24, 200, 24)
    pdf.ln(2)
    
    # Two-column layout for patient info and vital signs
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(190, 6, "Patient Information & Vital Signs", ln=True, border=1)
    
    # First row
    pdf.set_font("Arial", '', 8)
    pdf.cell(47.5, 5, f"Age: {patient_data['Age']}", 1)
    pdf.cell(47.5, 5, f"Gender: {patient_data['Gender']}", 1)
    pdf.cell(47.5, 5, f"Blood Group: {patient_data['Blood_Group']}", 1)
    pdf.cell(47.5, 5, f"Weight: {patient_data['Weight_kg']} kg", 1, ln=True)
    
    # Second row
    pdf.cell(47.5, 5, f"Temp: {patient_data['Temperature_C']}°C", 1)
    pdf.cell(47.5, 5, f"Heart Rate: {patient_data['Heart_Rate']} BPM", 1)
    pdf.cell(47.5, 5, f"BP: {patient_data['BP_Systolic']}/- mmHg", 1)
    pdf.cell(47.5, 5, f"Glucose: {patient_data['Glucose_Level']} mg/dL", 1, ln=True)
    
    # Symptoms in a single row
    conditions = []
    if patient_data['Has_Fever']: conditions.append("Fever")
    if patient_data['Has_Cough']: conditions.append("Cough")
    if patient_data['Has_Fatigue']: conditions.append("Fatigue")
    if patient_data['Has_Pain']: conditions.append("Pain")
    if patient_data['Has_Hypertension']: conditions.append("Hypertension")
    if patient_data['Has_Diabetes']: conditions.append("Diabetes")
    
    condition_text = ", ".join(conditions) if conditions else "None"
    pdf.cell(190, 5, f"Symptoms: {condition_text}", 1, ln=True)
    
    # Diagnosis Section
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(190, 6, "Diagnosis", ln=True, border=1)
    pdf.set_font("Arial", '', 9)
    
    # Diagnosis information in a more compact format
    pdf.cell(95, 5, f"Condition: {prediction_results.get('Predicted_Disease', 'Not available')}", 1)
    pdf.cell(47.5, 5, f"Risk: {prediction_results.get('Risk_Level', 'N/A')}", 1)
    pdf.cell(47.5, 5, f"Polypharmacy: {prediction_results.get('Polypharmacy_Risk', 'N/A')}", 1, ln=True)
    
    if prediction_results.get('Disease_Causes'):
        pdf.set_font("Arial", '', 8)
        # Limit text length to avoid overflow
        causes_text = prediction_results.get('Disease_Causes', '')[:150]
        if len(prediction_results.get('Disease_Causes', '')) > 150:
            causes_text += "..."
        pdf.cell(190, 5, f"Causes: {causes_text}", 1, ln=True)
    
    # Medication Section - more compact
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(190, 6, "Prescribed Medications", ln=True, border=1)
    pdf.set_font("Arial", '', 8)
    
    for i in range(1, 4):
        med_key = f'Medicine_{i}'
        dose_key = f'Dosage_{i}'
        freq_key = f'Frequency_{i}'
        dur_key = f'Duration_{i}'
        
        if prediction_results.get(med_key):
            med_name = prediction_results.get(med_key, '')
            dosage = prediction_results.get(dose_key, '')
            frequency = prediction_results.get(freq_key, '')
            duration = prediction_results.get(dur_key, '')
            pdf.cell(60, 5, f"{i}. {med_name}", 1)
            pdf.cell(40, 5, f"Dosage: {dosage}", 1)
            pdf.cell(45, 5, f"Freq: {frequency}", 1)
            pdf.cell(45, 5, f"Duration: {duration}", 1, ln=True)
    
    # Instructions and Health Tips - use shorter format
    pdf.ln(2)
    pdf.set_font("Arial", 'B', 10)
    pdf.cell(190, 6, "Instructions & Recommendations", ln=True, border=1)
    pdf.set_font("Arial", '', 8)
    
    # Get instruction values
    instr1 = prediction_results.get('Instructions_1', '')
    instr2 = prediction_results.get('Instructions_2', '')
    instr3 = prediction_results.get('Instructions_3', '')
    
    # Show instructions in compact format
    if instr1 or instr2 or instr3:
        instr_text = ""
        if instr1: instr_text += f"1. {instr1[:80]}... " if len(instr1) > 80 else f"1. {instr1} "
        if instr2: instr_text += f"2. {instr2[:80]}... " if len(instr2) > 80 else f"2. {instr2} "
        if instr3: instr_text += f"3. {instr3[:80]}... " if len(instr3) > 80 else f"3. {instr3} "
        pdf.multi_cell(190, 5, instr_text, 1)
    
    # Health tips (condensed)
    if prediction_results.get('Personalized_Health_Tips'):
        tips = prediction_results.get('Personalized_Health_Tips', '')[:150]
        if len(prediction_results.get('Personalized_Health_Tips', '')) > 150:
            tips += "..."
        pdf.cell(190, 5, f"Health Tips: {tips}", 1, ln=True)
    
    # Required Tests Section (condensed)
    if prediction_results.get('Required_Tests'):
        tests = ", ".join(prediction_results.get('Required_Tests', []))
        pdf.cell(190, 5, f"Recommended Tests: {tests}", 1, ln=True)
    
    # Add footer with disclaimer
    pdf.set_y(-20)
    pdf.set_font("Arial", 'I', 6)
    pdf.cell(190, 4, "Disclaimer: This is an AI-generated medical prediction and should be reviewed by a healthcare professional.", 0, ln=True, align='C')
    pdf.cell(190, 4, "Not for clinical use without professional medical consultation.", 0, ln=True, align='C')
    
    return pdf.output(dest="S").encode("latin1")

# Load the trained model and encoders
@st.cache_resource
def load_resources():
    # Change paths to load from the current directory
    pipeline = load_model('synthetic_v2_pipeline.joblib')
    target_encoders = joblib.load('synthetic_v2_target_encoders.joblib')
    mappings = joblib.load('synthetic_v2_disease_mappings.joblib')
    return pipeline, target_encoders, mappings

pipeline, target_encoders, mappings = load_resources()

# Sidebar for app information
with st.sidebar:
    st.image("https://img.icons8.com/plasticine/100/000000/hospital-3.png", width=100)
    st.markdown("## About")
    st.markdown("This application predicts medical outcomes based on patient data using a machine learning model.")
    st.markdown("### Instructions")
    st.markdown("1. Enter patient information in the form")
    st.markdown("2. Click 'Generate Prediction'")
    st.markdown("3. View the predicted outcome and recommendations")
    
    st.markdown("---")
    st.markdown("### 🔍 Model Information")
    st.markdown("Synthetic Medical Outcome Predictor")
    st.markdown("Version: 2.0")

# Main content
st.markdown('<p class="main-header">Medical Outcome Predictor</p>', unsafe_allow_html=True)

# Input features in a card with tabs
with st.container():
    st.markdown('<p class="sub-header">Patient Information</p>', unsafe_allow_html=True)
    
    tabs = st.tabs(["Demographics", "Symptoms", "Vital Signs"])
    
    with tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            patient_name = st.text_input("Patient Name", value="")
            age = st.number_input("Age", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender", options=["Male", "Female"])
        with col2:
            blood_group = st.selectbox("Blood Group", options=["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])
            weight = st.number_input("Weight (kg)", min_value=0.0, value=70.0, step=0.1)
    
    with tabs[1]:
        col1, col2 = st.columns(2)
        with col1:
            has_fever = st.checkbox("Has Fever")
            has_cough = st.checkbox("Has Cough")
            has_fatigue = st.checkbox("Has Fatigue")
        with col2:
            has_pain = st.checkbox("Has Pain")
            has_hypertension = st.checkbox("Has Hypertension")
            has_diabetes = st.checkbox("Has Diabetes")
    
    with tabs[2]:
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.number_input("Temperature (°C)", min_value=30.0, max_value=42.0, value=37.0, step=0.1)
            heart_rate = st.number_input("Heart Rate", min_value=30, max_value=200, value=70)
            bp_systolic = st.number_input("Blood Pressure (Systolic)", min_value=50, max_value=200, value=120)
        with col2:
            wbc_count = st.number_input("WBC Count", min_value=0.0, value=7.0, step=0.1)
            glucose_level = st.number_input("Glucose Level", min_value=0.0, value=90.0, step=0.1)

# Prepare input data for prediction
input_data = {
    'Patient_Name': patient_name,
    'Age': age,
    'Gender': gender,
    'Blood_Group': blood_group,
    'Weight_kg': weight,
    'Has_Fever': int(has_fever),
    'Has_Cough': int(has_cough),
    'Has_Fatigue': int(has_fatigue),
    'Has_Pain': int(has_pain),
    'Has_Hypertension': int(has_hypertension),
    'Has_Diabetes': int(has_diabetes),
    'Temperature_C': temperature,
    'Heart_Rate': heart_rate,
    'BP_Systolic': bp_systolic,
    'WBC_Count': wbc_count,
    'Glucose_Level': glucose_level
}

# Define the order of input features
input_feature_names = [
    'Patient_Name', 'Age', 'Gender', 'Blood_Group', 'Weight_kg',
    'Has_Fever', 'Has_Cough', 'Has_Fatigue', 'Has_Pain',
    'Has_Hypertension', 'Has_Diabetes', 'Temperature_C',
    'Heart_Rate', 'BP_Systolic', 'WBC_Count', 'Glucose_Level'
]

# Button to make predictions with a progress indicator
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("Generate Prediction", type="primary", use_container_width=True)

if predict_button:
    with st.spinner("Analyzing patient data..."):
        # Prepare feature values in the correct order
        feature_values = [input_data[name] for name in input_feature_names]
        encoded_input = encode_input_features(input_feature_names, feature_values)
        
        # Make predictions using the pipeline's predict function
        predictions_array = pipeline.predict(encoded_input)
        
        # Define the output columns that our model predicts
        target_columns = [
            'Predicted_Disease', 'Medicine_1', 'Dosage_1', 'Frequency_1', 'Duration_1',
            'Medicine_2', 'Dosage_2', 'Frequency_2', 'Duration_2',
            'Medicine_3', 'Dosage_3', 'Frequency_3', 'Duration_3',
            'Polypharmacy_Risk'
        ]
        
        # Convert predictions array to DataFrame
        predictions_encoded = pd.DataFrame(predictions_array, columns=target_columns)
        
        # Decode the predictions using target_encoders
        predictions_decoded = {}
        for col in predictions_encoded.columns:
            encoder = target_encoders.get(col)
            if encoder:
                predictions_decoded[col] = encoder.inverse_transform([predictions_encoded[col].iloc[0]])[0]
            else:
                predictions_decoded[col] = predictions_encoded[col].iloc[0]
        
        # Get mapping outputs based on the predicted disease
        predicted_disease = predictions_decoded.get('Predicted_Disease')
        mapping_outputs = get_mapped_outputs(predicted_disease, mappings)
        
        # Combine prediction outputs and mapping outputs into a single dictionary
        final_output = {**predictions_decoded, **mapping_outputs}
    
        # Display the prediction results in a more visually appealing way
    st.markdown('<p class="prediction-header">Diagnosis & Treatment Plan</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"### Diagnosis")
        st.markdown(f"**Condition:** {final_output.get('Predicted_Disease', 'Not available')}")
        st.markdown(f"**Risk Level:** {mapping_outputs.get('Risk_Level', 'Not available')}")
        st.markdown(f"**Polypharmacy Risk:** {final_output.get('Polypharmacy_Risk', 'Not available')}")
        
        # Add Disease Causes
        if final_output.get('Disease_Causes'):
            st.markdown("### Disease Causes")
            st.markdown(f"{final_output.get('Disease_Causes', 'Not available')}")
        st.markdown('</div>', unsafe_allow_html=True)
        
        if 'Required_Tests' in mapping_outputs:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### Recommended Tests")
            for test in mapping_outputs.get('Required_Tests', []):
                st.markdown(f"- {test}")
            st.markdown('</div>', unsafe_allow_html=True)
            
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### Prescribed Medications")
        
        if final_output.get('Medicine_1'):
            st.markdown(f"**1. {final_output.get('Medicine_1', '')}**")
            st.markdown(f"   - Dosage: {final_output.get('Dosage_1', '')}")
            st.markdown(f"   - Frequency: {final_output.get('Frequency_1', '')}")
            st.markdown(f"   - Duration: {final_output.get('Duration_1', '')}")
            
        if final_output.get('Medicine_2'):
            st.markdown(f"**2. {final_output.get('Medicine_2', '')}**")
            st.markdown(f"   - Dosage: {final_output.get('Dosage_2', '')}")
            st.markdown(f"   - Frequency: {final_output.get('Frequency_2', '')}")
            st.markdown(f"   - Duration: {final_output.get('Duration_2', '')}")
            
        if final_output.get('Medicine_3'):
            st.markdown(f"**3. {final_output.get('Medicine_3', '')}**")
            st.markdown(f"   - Dosage: {final_output.get('Dosage_3', '')}")
            st.markdown(f"   - Frequency: {final_output.get('Frequency_3', '')}")
            st.markdown(f"   - Duration: {final_output.get('Duration_3', '')}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # New card for specific instructions
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Specific Instructions")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if final_output.get('Instructions_1'):
            st.markdown(f"**Instruction 1:**")
            st.markdown(f"{final_output.get('Instructions_1', 'None')}")
    with col2:
        if final_output.get('Instructions_2'):
            st.markdown(f"**Instruction 2:**")
            st.markdown(f"{final_output.get('Instructions_2', 'None')}")
    with col3:
        if final_output.get('Instructions_3'):
            st.markdown(f"**Instruction 3:**")
            st.markdown(f"{final_output.get('Instructions_3', 'None')}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional information in expanders
    with st.expander("Personalized Health Tips"):
        if final_output.get('Personalized_Health_Tips'):
            st.markdown(f"{final_output.get('Personalized_Health_Tips')}")
        else:
            st.write("No personalized health tips available.")
            
    with st.expander("Polypharmacy Recommendation"):
        if final_output.get('Polypharmacy_Recommendation'):
            st.markdown(f"{final_output.get('Polypharmacy_Recommendation')}")
        else:
            st.write("No polypharmacy recommendations available.")
    
    # Raw JSON output for reference (collapsed by default)
    with st.expander("View Raw Prediction Data"):
        st.json(final_output)
        
    # Add PDF download section with a visual separator
    st.markdown('<hr style="margin-top: 30px; margin-bottom: 30px;">', unsafe_allow_html=True)
    
    # Create a centered container for the download button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("### 📄 Download Complete Medical Report")
        st.markdown("Get a professionally formatted medical report with all diagnosis and treatment details.")
        
        # Generate PDF report
        pdf_bytes = generate_pdf_report(input_data, final_output)
        
        # Create download button
        patient_name_safe = patient_name.replace(" ", "_") if patient_name else "Patient"
        st.download_button(
            label="Download Medical Report (PDF)",
            data=pdf_bytes,
            file_name=f"Medical_Report_{patient_name_safe}_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )