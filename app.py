# app.py
import streamlit as st
import pandas as pd
import numpy as np
# Ensure these imports point to the correct, updated logic and mappings
from prediction_calculator_logic import get_prediction, ALL_MODEL_FEATURES
from question_mappings import QUESTION_MAPPINGS

# --- Page Configuration ---
st.set_page_config(page_title="Child Behavioral Insights Estimator", layout="wide")

# --- Custom CSS for Styling ---
st.markdown(""" <style>
/* Define color palette */
:root {
    --primary-color: #1e3a8a; /* Dark Blue */
    --secondary-color: #60a5fa; /* Light Blue */
    --background-color: #f0f4f8; /* Light Grey-Blue */
    --text-color-dark: #1f2937; /* Dark Gray */
    --text-color-light: #f9fafb; /* Very Light Gray / White */
    --accent-color: #f59e0b; /* Amber */
    --disclaimer-bg-color: #e0e7ff; /* Light Blue/Lavender for disclaimer box */
    --progress-bar-color: #3b82f6; /* Medium Blue for progress bar */
    --results-button-bg: #ffffff; /* White background for results buttons */
    --results-button-border: #d1d5db; /* Gray border for results buttons */
    --results-button-hover-bg: #f3f4f6; /* Light gray hover for results buttons */
}

/* Apply background color */
.stApp {
    background-color: var(--background-color);
}

/* Default text color for the app */
body, .stApp, .stMarkdown, .stRadio > label,
.stTextInput > div > div > input {
    color: var(--text-color-dark) !important;
}

/* Style Headers */
h1 { /* Main Title */
    color: var(--primary-color);
    text-align: center;
    margin-bottom: 1rem;
}
h2, h3 { /* General Subheaders */
    color: var(--primary-color);
    text-align: center;
    font-weight: 600;
}

/* Centering and styling for intro text container */
.intro-text-container {
    text-align: left;
    color: var(--text-color-dark);
    background-color: #ffffff;
    padding: 1.5rem;
    border-radius: 8px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    max-width: 750px;
    margin: 1rem auto 2rem auto;
}
.intro-text-container h2 { /* "Understanding Your Child's..." */
    text-align: left;
    margin-bottom: 0.75rem;
    font-size: 1.5rem;
}
.intro-text-container h3 { /* "Important Considerations:" */
    text-align: left;
    font-size: 1.2rem;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}
.intro-text-container p, .intro-text-container li {
    font-size: 1rem;
    line-height: 1.6;
}
.intro-text-container strong { /* Bold text within intro */
     color: var(--primary-color);
}

/* --- Questionnaire Styling --- */

/* Ensure questions within the form are centered */
div[data-testid="stForm"] div[data-testid="stMarkdownContainer"] p {
    text-align: center !important; /* Force centering for question text */
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    font-weight: 500;
}
/* Bold text within questions */
div[data-testid="stForm"] div[data-testid="stMarkdownContainer"] p strong {
    font-weight: bold;
    color: var(--text-color-dark);
}

/* Style Radio buttons */
.stRadio [role="radiogroup"] {
    display: flex;
    justify-content: center;
    width: 100%;
    color: var(--text-color-dark) !important;
    margin-bottom: 0.5rem;
}
.stRadio > label { /* Container for radio button */
    background-color: transparent !important;
}
.stRadio label span { /* Text label for radio button option */
    color: var(--text-color-dark) !important;
    padding: 0 5px;
}

/* Style the Form Submit Button (Get Estimate) */
div[data-testid="stFormSubmitButton"] button {
    display: block;
    margin: 1.5rem auto 0 auto;
    background-color: var(--primary-color);
    color: var(--text-color-light) !important; /* Ensure light text */
    border: none;
    padding: 0.75rem 1.5rem;
    border-radius: 0.375rem;
    font-weight: 600;
    width: auto;
}
div[data-testid="stFormSubmitButton"] button:hover {
    background-color: var(--secondary-color);
    color: var(--text-color-dark) !important;
}

/* --- Results Page Specific Styling --- */

/* Centering wrapper for results content */
.results-column {
    text-align: center; /* Center align block elements like h2, p, divs */
    padding-top: 1rem; /* Add some padding at the top, but not excessive */
}

/* Main Risk Title (e.g., "Higher Risk") - Updated class name */
.results-column .estimation-title { /* Changed from h2 directly */
    font-size: 2.2rem; /* Slightly adjusted size */
    font-weight: 700;
    color: var(--text-color-dark); /* Dark text */
    text-align: center;
    margin-top: 1rem;
    margin-bottom: 0.5rem;
}

/* Bottom Line Summary Text */
.bottom-line-summary {
    font-size: 1.1rem;
    font-weight: 500;
    color: var(--text-color-dark);
    margin-top: 0rem;
    margin-bottom: 1.5rem;
    text-align: center;
    max-width: 600px; /* Limit width for readability */
    margin-left: auto;
    margin-right: auto;
}

/* Probability Section - Block is centered, text inside left-aligned */
.probability-section {
    max-width: 400px;
    margin: 0 auto 1.5rem auto; /* margin:auto centers the block */
    text-align: left;
}
.probability-title {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-color-dark);
    text-align: left;
    margin-bottom: 0.25rem;
}
.probability-value {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--primary-color);
    text-align: left;
    margin-top: 0.25rem;
}
/* Style the progress bar color */
.stProgress > div > div > div > div {
    background-color: var(--progress-bar-color) !important;
}

/* Disclaimer Box - Block is centered, text inside centered */
.disclaimer-box {
    background-color: var(--disclaimer-bg-color);
    color: #4b5563;
    padding: 1rem;
    border-radius: 8px;
    margin-top: 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    font-size: 0.9rem;
    border: 1px solid var(--secondary-color);
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}
.disclaimer-box strong {
    color: var(--text-color-dark);
    font-weight: 600;
}


/* Recommendations Section - Block elements centered, text inside left-aligned */
.recommendations-section {
    text-align: center;
}
.recommendations-section h3 {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--primary-color);
    text-align: center;
    margin-top: 2rem;
    margin-bottom: 1rem;
}
.recommendations-section p, .recommendations-section li {
    text-align: left;
    font-size: 1rem;
    line-height: 1.6;
    color: var(--text-color-dark);
    max-width: 600px;
    margin-left: auto;
    margin-right: auto;
}
.recommendations-section strong {
    font-weight: 600;
     color: var(--text-color-dark);
}
.recommendations-section em {
    font-style: italic;
    color: #4b5563;
}
.recommendations-section ul {
    display: inline-block;
    text-align: left;
    margin-top: 0.5rem;
    padding-left: 20px;
}


/* Results Page Buttons Container */
.results-buttons-container {
    text-align: center; /* Center the buttons within this div */
    margin-top: 1.5rem;
    margin-bottom: 1rem;
}

/* Results Page Buttons (Generate Rec, Start Again, Hide Rec) */
.results-buttons-container .stButton button {
     display: inline-block; /* Keep inline-block */
     margin: 0.5rem 0.5rem; /* Adjusted margin slightly */
     background-color: var(--results-button-bg) !important;
     color: var(--text-color-dark) !important;
     border: 1px solid var(--results-button-border) !important;
     padding: 0.5rem 1rem;
     border-radius: 0.375rem;
     font-weight: 600;
     width: auto;
     line-height: 1.5;
     /* Remove specific column centering styles as the container handles it */
}
.results-buttons-container .stButton button:hover {
    background-color: var(--results-button-hover-bg) !important;
    color: var(--text-color-dark) !important;
    border-color: #9ca3af !important;
}


/* Hr separator */
hr {
    border-top: 1px solid #ccc;
}

</style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
def init_session_state():
    # ... (session state initialization remains the same) ...
    if 'answers' not in st.session_state:
        st.session_state.answers = {}
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False
    if 'prediction_class' not in st.session_state:
        st.session_state.prediction_class = None
    if 'prediction_probs' not in st.session_state:
        st.session_state.prediction_probs = None
    if 'show_recommendation' not in st.session_state:
        st.session_state.show_recommendation = False

init_session_state()

# --- Reset Function ---
def start_again():
    # ... (reset function remains the same) ...
    keys_to_reset = ['answers', 'show_results', 'prediction_class', 'prediction_probs', 'show_recommendation']
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]
    init_session_state()
    st.rerun()

# --- Main App ---
st.title("Child Behavioral Insights Estimator")

# --- Introductory Text Section ---
st.markdown("""
<div class="intro-text-container">
    <h2>Understanding Your Child's Behavioral Patterns</h2>
    <p>This tool is designed to provide an <strong>estimation</strong> based on observed behavioral patterns, utilizing statistical insights derived from the comprehensive ABCD (Adolescent Brain Cognitive Development) Study. By answering the questions below regarding your child's behavior over the <strong>past six months</strong>, you can gain a data-informed perspective.</p>
    <h3>Important Considerations:</h3>
    <ul>
        <li>This instrument serves as an <strong>estimation tool only</strong> and <strong>does not constitute a medical diagnosis</strong>.</li>
        <li>The insights are generated from group-level statistical data and may not be specifically applicable to every individual's unique circumstances.</li>
        <li>It is imperative to <strong>consult with a qualified healthcare professional</strong> for any concerns related to your child's health, development, or well-being.</li>
    </ul>
    <p>Please proceed to the questionnaire to begin.</p>
</div>
""", unsafe_allow_html=True)


# --- Questionnaire Section ---
if not st.session_state.show_results:
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        with st.form(key='prediction_form'):
            st.subheader("Parent Questionnaire")
            # QUESTION_MAPPINGS is imported from question_mappings.py
            # The order of questions will be determined by the order in that list.
            
            for i, question_map in enumerate(QUESTION_MAPPINGS):
                feature_id = question_map['id']
                question_text = question_map['question_text']
                options_dict = question_map['options'] # For radio, this is {label: value}; for continuous, {min, max, default, step}
                scale_type = question_map['scale_type']

                st.markdown(f"{i+1}. {question_text}", unsafe_allow_html=True)

                if scale_type == 'Continuous':
                    min_val = options_dict.get('min_value', 11)
                    max_val = options_dict.get('max_value', 55)
                    # Get current answer from session_state.answers or use default from options_dict
                    current_answer_val = st.session_state.answers.get(feature_id, options_dict.get('default_value', (min_val + max_val) // 2))
                    
                    selected_value = st.number_input(
                        label=f"Enter value for {feature_id}", # Label is collapsed
                        min_value=min_val,
                        max_value=max_val,
                        value=int(current_answer_val), # Ensure value is int if loaded from session
                        step=options_dict.get('step', 1),
                        key=f"{feature_id}_input", # Unique key for number_input
                        label_visibility="collapsed"
                    )
                    # Store in a temporary session state key that matches the input widget to retrieve later
                    # This is handled by st.form automatically via the key.
                else: # Handles 'Binary', '3-point', '5-point' with st.radio
                    radio_options_labels = list(options_dict.keys())
                    # Get current *value* from session_state.answers to find the *label* for index
                    current_answer_value = st.session_state.answers.get(feature_id)
                    current_radio_index = None
                    if current_answer_value is not None:
                        for label, val in options_dict.items():
                            if val == current_answer_value:
                                if label in radio_options_labels:
                                    current_radio_index = radio_options_labels.index(label)
                                break
                    
                    selected_label = st.radio(
                        label=f"Select an option for {feature_id}", # Label is collapsed
                        options=radio_options_labels,
                        key=f"{feature_id}_radio", # Unique key for radio
                        horizontal=True,
                        label_visibility="collapsed",
                        index=current_radio_index # Pre-select if already answered
                    )

                if i < len(QUESTION_MAPPINGS) - 1:
                    st.markdown("<hr style='margin-top: 0.5rem; margin-bottom: 0.5rem;'>", unsafe_allow_html=True)

            submitted = st.form_submit_button("Get Estimate")

            if submitted:
                collected_answers = {}
                all_answered = True
                for question_map in QUESTION_MAPPINGS: # Iterate again to collect based on type
                    feature_id = question_map['id']
                    options_dict = question_map['options']
                    scale_type = question_map['scale_type']

                    if scale_type == 'Continuous':
                        # Value from number_input is directly its numerical value
                        val_from_input = st.session_state.get(f"{feature_id}_input")
                        if val_from_input is not None: # Should always have a value from number_input
                            collected_answers[feature_id] = val_from_input
                        else:
                            # This case should ideally not be hit if number_input has a default
                            all_answered = False 
                            st.warning(f"Please ensure a value is entered for: {question_map['question_text']}", icon="⚠️")
                            break
                    else: # Radio button based inputs
                        selected_label = st.session_state.get(f"{feature_id}_radio")
                        if selected_label is not None:
                            collected_answers[feature_id] = options_dict[selected_label] # Map label back to value
                        else:
                            all_answered = False # A radio option was not selected
                            st.warning(f"Please answer question: {question_map['question_text']}", icon="⚠️")
                            break
                
                if not all_answered:
                    st.session_state.answers = collected_answers # Store partially collected answers for pre-filling
                    st.warning("Please answer all questions before submitting.", icon="⚠️")
                    # No rerun here, let user fix on the same page
                else:
                    st.session_state.answers = collected_answers
                    # Assuming get_prediction can handle the raw numerical values collected
                    pred_class, pred_probs = get_prediction(st.session_state.answers)
                    if pred_class is not None and pred_probs is not None:
                        st.session_state.prediction_class = pred_class
                        st.session_state.prediction_probs = pred_probs
                        st.session_state.show_results = True
                        st.rerun()
                    else:
                        st.error("Could not retrieve prediction. Please ensure the model is loaded correctly.")

# --- Results Section ---
if st.session_state.show_results:
    res_col1, res_col2, res_col3 = st.columns([0.5, 3, 0.5])
    with res_col2:
        st.markdown(f'<div class="results-column">', unsafe_allow_html=True)

        pred_class = st.session_state.prediction_class
        pred_probs = st.session_state.prediction_probs
        # ---- MODIFICATION: Changed Risk Phrasing ----
        if pred_class == 1:
            estimation_title = "Estimation suggests patterns consistent with Depression"
            bottom_line_summary = "Based on the patterns observed, these results suggest a potential alignment with depressive symptoms. Consultation with a healthcare professional is strongly recommended for a formal evaluation."
        else:
            estimation_title = "Estimation suggests patterns less consistent with Depression"
            bottom_line_summary = "The observed patterns currently align with a lower likelihood of depression based on this estimation. Continued monitoring and open communication remain important."
        # -----------------------------------------------

        # Display estimation title
        # ---- MODIFICATION: Use a specific class for easier styling ----
        st.markdown(f'<h2 class="estimation-title">{estimation_title}</h2>', unsafe_allow_html=True)
        # -------------------------------------------------------------

        # Display Bottom Line Summary
        st.markdown(f'<p class="bottom-line-summary">{bottom_line_summary}</p>', unsafe_allow_html=True)


        # Display Only Highest Probability
        if pred_probs:
            max_prob = max(pred_probs)
            max_index = pred_probs.index(max_prob)
            # ---- MODIFICATION: Changed Probability Title Phrasing ----
            prob_title = "Your Child's Risk of Depression:" if max_index == 1 else "Your Child's Risk of Depression:"
            # -------------------------------------------------------

            # Centered probability display block
            st.markdown('<div class="probability-section">', unsafe_allow_html=True)
            st.markdown(f'<p class="probability-title">{prob_title}</p>', unsafe_allow_html=True)
            st.progress(float(max_prob))
            st.markdown(f'<p class="probability-value">{max_prob*100:.2f}%</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)


        # Disclaimer box
        st.markdown("""
        <div class="disclaimer-box">
            <strong>Disclaimer:</strong> This prediction is based on a statistical model and is <strong>not</strong> a medical diagnosis or a substitute for professional advice.
            The estimation reflects patterns compared to data from the ABCD study, where the threshold for 'Higher Risk' patterns was set based on the 75th percentile of a depression score measure.
            Individual results may vary. Please consult with a qualified healthcare professional for any health concerns.
        </div>
        """, unsafe_allow_html=True) # Updated disclaimer text slightly for clarity

        # --- MODIFICATION: Centralized Button Area ---
        st.markdown('<div class="results-buttons-container">', unsafe_allow_html=True)

        # Recommendation Section (show/hide logic inside the container)
        if st.session_state.show_recommendation:
             # Display recommendations (conditionally based on pred_class)
             # ... (recommendation text rendering logic remains the same) ...
             st.markdown('<h3>Further Information & Recommendations</h3>', unsafe_allow_html=True)
             st.markdown('<div class="recommendations-section" style="text-align: left; max-width: 600px; margin: 1rem auto;">', unsafe_allow_html=True) # Added inline style for text alignment and width

             if pred_class == 1:
                 st.markdown("""
                 <p>Based on your responses, some patterns suggest a potential for higher risk. It is advisable to consult with a
                 healthcare professional for a comprehensive evaluation. They can provide personalized advice and support.</p>
                 <p><strong>For further reading, consider these resources:</strong></p>
                 <ul>
                     <li>National Institute of Mental Health (NIMH): <a href="https://www.nimh.nih.gov/health/topics/depression" target="_blank">NIMH Depression Information</a></li>
                     <li>Anxiety & Depression Association of America (ADAA): <a href="https://adaa.org/understanding-anxiety/depression" target="_blank">ADAA Depression Information</a></li>
                 </ul>
                 <p><em>This information is not a substitute for professional medical advice.</em></p>
                 """, unsafe_allow_html=True)
             else:
                 st.markdown("""
                 <p>Your responses generally indicate a lower risk profile according to this model. Maintaining healthy habits, open communication,
                 and continued observation are always beneficial for mental well-being.</p>
                  <p><strong>For general well-being tips, you might find these helpful:</strong></p>
                 <ul>
                     <li>MentalHealth.gov: <a href="https://www.mentalhealth.gov/basics/what-is-mental-health" target="_blank">What is Mental Health?</a></li>
                     <li>Active Minds: <a href="https://www.activeminds.org/about-mental-health/self-care/" target="_blank">Self-Care Resources</a></li>
                 </ul>
                 <p><em>This information is not a substitute for professional medical advice.</em></p>
                 """, unsafe_allow_html=True)
             st.markdown('</div>', unsafe_allow_html=True) # Close recommendations-section div

             # Hide Recommendation Button (inside container)
             if st.button("Hide Recommendation", key="hide_rec_btn"):
                 st.session_state.show_recommendation = False
                 st.rerun()
        else:
             # Generate Recommendation Button (inside container)
             if st.button("Generate Recommendation", key="show_rec_btn"):
                 st.session_state.show_recommendation = True
                 st.rerun()

        # Start Again Button (inside container)
        if st.button("Start Questionnaire Again", key="start_again_btn"):
            start_again()

        st.markdown('</div>', unsafe_allow_html=True) # Close results-buttons-container
        # --- END OF MODIFICATION ---

        st.write("---") # Horizontal line

        st.markdown('</div>', unsafe_allow_html=True) # Close .results-column

# --- Footer Placeholder (Optional) ---
# st.markdown("<div style='text-align: center; margin-top: 2rem; color: grey;'>App Footer</div>", unsafe_allow_html=True)
