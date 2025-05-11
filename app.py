import streamlit as st
from prediction_calculator_logic import get_prediction, RF_MODEL, ALL_MODEL_FEATURES # DEFAULT_SELECTABLE_FEATURES no longer directly used here
from question_mappings import QUESTION_MAPPINGS
# import pandas as pd # No longer strictly needed here

# --- Color Palette ---
COLOR_PRIMARY = "#2B3A67"  # A deep blue
COLOR_SECONDARY = "#4A5D8F" # A lighter, softer blue
COLOR_ACCENT = "#F7B538"    # A warm yellow/orange for accents if needed
COLOR_BACKGROUND_LIGHT = "#F0F2F6" # Light grey for backgrounds
COLOR_TEXT_LIGHT = "#FFFFFF"
COLOR_TEXT_DARK = "#333333"
COLOR_BORDER = "#D1D9E6" # Softer border color

def local_css():
    st.markdown(f"""
    <style>
    .main-container {{
        max-width: 800px; /* Or your preferred max width */
        margin: auto;
        padding: 20px;
        background-color: {COLOR_BACKGROUND_LIGHT}; /* Optional: background for the whole centered area */
        border-radius: 10px; /* Optional: rounded corners for the main centered area */
    }}
    .stApp {{
        background-color: {COLOR_BACKGROUND_LIGHT}; /* Set a global background for the app */
        color: {COLOR_TEXT_DARK}; /* Default dark text color for the app */
    }}
    body {{
        color: {COLOR_TEXT_DARK}; /* Ensure body text is dark */
    }}
    h1 {{
        color: {COLOR_PRIMARY};
        text-align: center;
    }}
    h2, h3 {{ /* Headers for Questionnaire, Results etc. */
        color: {COLOR_SECONDARY};
        text-align: center; /* Center other headers too */
    }}
    .questionnaire-box {{
        border: 2px solid {COLOR_BORDER};
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        margin-bottom: 20px;
        background-color: {COLOR_TEXT_LIGHT}; /* White background for the box */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* Subtle shadow */
        color: {COLOR_TEXT_DARK}; /* Ensure text inside the box is dark */
    }}
    .question-text {{
        text-align: center !important; /* Center question text */
        color: {COLOR_TEXT_DARK} !important; /* Ensure question text is dark */
        margin-bottom: 5px; /* Add some space below question text */
    }}
    /* Styling for Streamlit Radio Buttons */
    div[data-testid="stRadio"] > label > div[data-testid="stMarkdownContainer"] > p {{
        color: {COLOR_TEXT_DARK} !important; /* Ensures the main label of radio is dark if ever made visible */
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] {{ /* Targets the group of radio options */
        display: flex !important; 
        justify-content: center !important; /* Centers the Yes/No options */
        width: 100%; /* Ensure it takes full width for centering */
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label {{ /* Targets each individual Yes/No label container */
        background-color: transparent !important; /* Remove any unwanted background */
        margin: 0 5px !important; /* Adjust spacing */
        padding: 5px !important; /* Add some padding inside the label */
    }}
    div[data-testid="stRadio"] div[role="radiogroup"] > label span,
    div[data-testid="stRadio"] div[role="radiogroup"] > label p {{ /* Targets the text part of Yes/No */
        color: {COLOR_TEXT_DARK} !important; /* Makes "Yes" / "No" text dark */
        background-color: transparent !important; /* Ensure text background is also transparent */
    }}
    .stButton>button {{
        /* background-color: {COLOR_SECONDARY}; */ /* Example: keep default or use your color */
        color: {COLOR_TEXT_LIGHT} !important; /* Ensure button text is light if background is dark */
        /* If button background is light, use COLOR_TEXT_DARK for text */
        /* border-radius: 5px; */
        /* border: none; */
        /* padding: 10px 15px; */
        display: block !important; /* Make button a block to center it */
        margin-left: auto !important;
        margin-right: auto !important;
    }}
    /* Adjust submit button text color specifically if it has a dark background by default */
    div[data-testid="stFormSubmitButton"] button {{
        background-color: {COLOR_SECONDARY} !important; /* Example: Making it match secondary color */
        color: {COLOR_TEXT_LIGHT} !important; /* Light text on dark button */
    }}
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Depression Risk Estimator", layout="centered") # Use Streamlit's centered layout
    local_css()

    # Wrap main content in a div for centering and styling (if not using st.set_page_config's centered layout directly)
    # st.markdown('<div class="main-container">', unsafe_allow_html=True) # This line is optional if layout="centered" is enough

    st.title("Depression Risk Estimator")

    # Initialize session state variables if they don't exist
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'prediction_class' not in st.session_state:
        st.session_state.prediction_class = None
    if 'prediction_probs' not in st.session_state:
        st.session_state.prediction_probs = None
    if 'show_recommendation' not in st.session_state: # For Prompt 5
        st.session_state.show_recommendation = False

    st.markdown("""
    This tool provides an **experimental** estimation of depression risk based on your answers to the following questions. 
    The underlying model is a Random Forest classifier.
    
    **Important:** This is a simplified model for demonstration and should **not** be used for actual medical diagnosis or decision-making.
    """, unsafe_allow_html=True) # Allow HTML for potential future styling in this markdown

    # Check if model and feature names were loaded correctly in the logic module
    if RF_MODEL is None or not ALL_MODEL_FEATURES:
        st.error("Model or essential configuration could not be loaded. Please check the backend logic and ensure necessary files are present. The application cannot proceed.")
        st.stop()

    if not st.session_state.show_results:
        st.markdown('<div class="questionnaire-box">', unsafe_allow_html=True)
        st.header("Questionnaire") # This will be centered by h2,h3 rule
        st.markdown("<p style='text-align:center;'>Please answer the following questions based on your observations:</p>", unsafe_allow_html=True)

        # Use a form for better user experience with multiple radio buttons
        with st.form(key='questionnaire_form'):
            for question_map in QUESTION_MAPPINGS:
                q_id = question_map['id']
                q_text = question_map['question_text']
                # Ensure each radio button has a default value from session state or None
                # The actual answer will be stored by Streamlit when the radio button is interacted with
                # and then explicitly put into st.session_state.answers on submit.
                st.markdown(f'<div class="question-text"><b>{q_text}</b></div>', unsafe_allow_html=True)
                # st.markdown(f"**{q_text}**") # Old way
                answer = st.radio(
                    label="Select your answer:", # This label is hidden anyway
                    options=['Yes', 'No'], 
                    key=f"radio_{q_id}", 
                    horizontal=True,
                    label_visibility="collapsed" # Hide the generic "Select your answer:" label
                )
                # Temporarily store in a different part of session state if needed or directly access on submit
                # For now, we'll retrieve them from form state on submit
            
            submit_button = st.form_submit_button(label='Submit Questionnaire')
        
        st.markdown('</div>', unsafe_allow_html=True) # Close questionnaire-box div

        if submit_button:
            # Collect answers from the form state
            current_answers = {}
            for question_map in QUESTION_MAPPINGS:
                q_id = question_map['id']
                current_answers[q_id] = st.session_state[f'radio_{q_id}'] # radio_ answewrs are in st.session_state directly
            
            st.session_state.answers = current_answers
            
            if not st.session_state.answers or len(st.session_state.answers) < len(QUESTION_MAPPINGS):
                st.warning("Please answer all questions before submitting.") # Or handle partial submissions if desired
            else:
                with st.spinner('Calculating...'):
                    predicted_class, probabilities = get_prediction(st.session_state.answers)
                    st.session_state.prediction_class = predicted_class
                    st.session_state.prediction_probs = probabilities
                    st.session_state.show_results = True
                    st.session_state.show_recommendation = False # Reset recommendation visibility
                    st.rerun() # Rerun to show results section
    else:
        # This section will be built out in Prompt 4 (Results Display & Start Again)
        st.header("Prediction Results") # This will be centered by h2,h3 rule
        if st.session_state.prediction_class is not None and st.session_state.prediction_probs is not None:
            risk_label = "Higher Risk" if st.session_state.prediction_class == 1 else "Lower Risk"
            st.metric(label="Estimated Risk Category", value=risk_label)
            
            prob_class_0 = st.session_state.prediction_probs[0]
            prob_class_1 = st.session_state.prediction_probs[1]

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Probability of Lower Risk (Class 0):**")
                st.progress(prob_class_0)
                st.write(f"{prob_class_0:.2%}")
            
            with col2:
                st.write("**Probability of Higher Risk (Class 1):**")
                st.progress(prob_class_1)
                st.write(f"{prob_class_1:.2%}")
            
            st.info("**Disclaimer:** This prediction is based on a statistical model and is not a medical diagnosis. The threshold for 'Higher Risk' was set at the 75th percentile of the depression score in the training data.")

            # --- Recommendation Feature (Prompt 5) ---
            if not st.session_state.show_recommendation:
                if st.button("Generate Recommendation", key="gen_reco"):
                    st.session_state.show_recommendation = True
                    st.rerun()
            else:
                st.subheader("Further Information & Recommendations")
                if st.session_state.prediction_class == 1: # Higher Risk
                    st.markdown("""
                    Based on your responses, some patterns suggest a potential for higher risk. It's advisable to consult with a 
                    healthcare professional for a comprehensive evaluation. They can provide personalized advice and support.
                    
                    **For further reading, consider these resources:**
                    *   National Institute of Mental Health (NIMH): [https://www.nimh.nih.gov/health/topics/depression](https://www.nimh.nih.gov/health/topics/depression)
                    *   Anxiety & Depression Association of America (ADAA): [https://adaa.org/understanding-anxiety/depression](https://adaa.org/understanding-anxiety/depression)
                    """)
                else: # Lower Risk
                    st.markdown("""
                    Your responses generally indicate a lower risk profile according to this model. Maintaining healthy habits 
                    is always beneficial for mental well-being.
                    
                    **For general well-being tips, you might find these helpful:**
                    *   MentalHealth.gov: [https://www.mentalhealth.gov/basics/what-is-mental-health](https://www.mentalhealth.gov/basics/what-is-mental-health)
                    *   Active Minds: [https://www.activeminds.org/about-mental-health/self-care/](https://www.activeminds.org/about-mental-health/self-care/)
                    """)
                st.markdown("_This information is not a substitute for professional medical advice._")
                if st.button("Hide Recommendation", key="hide_reco"):
                    st.session_state.show_recommendation = False
                    st.rerun()
            # --- End Recommendation Feature ---

        else:
            st.error("Could not retrieve a prediction. Please check the input values or backend logs.")
        
        # Placeholder for "Start Again" button (to be fully implemented in Prompt 4)
        if st.button("Start Again", key="start_again_results"):
            st.session_state.answers = {}
            st.session_state.show_results = False
            st.session_state.prediction_class = None
            st.session_state.prediction_probs = None
            st.session_state.show_recommendation = False # Reset recommendation visibility
            st.rerun()

    # st.markdown('</div>', unsafe_allow_html=True) # Close main-container div (if used)

if __name__ == "__main__":
    main() 