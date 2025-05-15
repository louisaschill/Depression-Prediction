import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ----------------------------------------
# 1. Setup
# ----------------------------------------
st.set_page_config(page_title="Child Depression Risk Calculator", layout="centered")
st.title("Child Depression Risk Calculator")
st.markdown("""
Please answer the following questions based on your child's behaviors in the past few weeks.

Use this scale:
- 0 = Never
- 1 = Sometimes
- 2 = Often
""")

# ----------------------------------------
# 2. Load cleaned baseline data
# ----------------------------------------
@st.cache_data
def load_baseline():
    df = pd.read_csv("filtered_merged_variables.csv")
    top_vars = [
        'ksads_sleepprob_raw_814_p', 'cbcl_q71_p', 'cbcl_q04_p', 
        'famhx_ss_parent_prf_p', 'sds_p_ss_does', 'cbcl_q86_p', 
        'cbcl_q09_p', 'asr_q59_p', 'asr_q47_p', 
        'cbcl_q112_p', 'sds_p_ss_total', 'cbcl_q22_p'
    ]
    return df[top_vars].dropna()

baseline_df = load_baseline()

# ----------------------------------------
# 3. Logistic Regression Coefficients
# ----------------------------------------
intercept = -0.7132

coefficients = {
    'ksads_sleepprob_raw_814_p': 0.1580,
    'cbcl_q71_p': 0.1264,
    'cbcl_q04_p': 0.1170,
    'famhx_ss_parent_prf_p': -0.1068,
    'sds_p_ss_does': 0.1030,
    'cbcl_q86_p': 0.1008,
    'cbcl_q09_p': 0.0986,
    'asr_q59_p': 0.0963,
    'asr_q47_p': 0.0947,
    'cbcl_q112_p': 0.0884,
    'sds_p_ss_total': 0.0876,
    'cbcl_q22_p': 0.0864
}

labels = {
    'ksads_sleepprob_raw_814_p': "Child has sleep problems (0=No, 1=Yes)",
    'cbcl_q71_p': "Child is self-conscious/easily embarrassed (0=Not True, 1=Somewhat, 2=Very True)",
    'cbcl_q04_p': "Child cries a lot (0=Not True, 1=Somewhat, 2=Very True)",
    'famhx_ss_parent_prf_p': "Parent has history of psychiatric problems (0=No, 1=Yes)",
    'sds_p_ss_does': "Child's social engagement (1=Never to 5=Always)",
    'cbcl_q86_p': "Child feels too guilty (0=Not True, 1=Somewhat, 2=Very True)",
    'cbcl_q09_p': "Child fears doing bad things (0=Not True, 1=Somewhat, 2=Very True)",
    'asr_q59_p': "Child has somatic complaints (e.g., aches) (0=Not True, 1=Somewhat, 2=Very True)",
    'asr_q47_p': "Child has anxiety symptoms (e.g., nervousness) (0=Not True, 1=Somewhat, 2=Very True)",
    'cbcl_q112_p': "Child worries a lot (0=Not True, 1=Somewhat, 2=Very True)",
    'sds_p_ss_total': "Child's total score of social responsiveness (11-55)",
    'cbcl_q22_p': "Child feels worthless or inferior (0=Not True, 1=Somewhat, 2=Very True)"
}

# ----------------------------------------
# 4. Inputs + Plots
# ----------------------------------------
responses = {}
for var in coefficients:
    label = labels.get(var, var)
    col1, col2 = st.columns([2, 3])

    with col1:
        if var == 'ksads_sleepprob_raw_814_p' or var == 'famhx_ss_parent_prf_p':
            val = st.selectbox(f"{label}", options=[0, 1], index=0, key=var)
        elif var == 'sds_p_ss_does':
            val = st.selectbox(f"{label}", options=[1, 2, 3, 4, 5], index=0, key=var)
        elif var == 'sds_p_ss_total':
            val = st.number_input(f"{label}", min_value=11, max_value=55, value=33, step=1, key=var)
        else:
            val = st.selectbox(f"{label}", options=[0, 1, 2], index=0, key=var)
        responses[var] = val

    with col2:
        fig, ax = plt.subplots()
        sns.histplot(baseline_df[var], bins='auto', ax=ax, kde=False)
        ax.axvline(val, color='red', linestyle='--', label="Your response")
        ax.set_title("ABCD Distribution")
        ax.set_xlabel("Response")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)

# ----------------------------------------
# 5. Risk Score
# ----------------------------------------
if st.button("Calculate Depression Risk"):
    log_odds = intercept + sum(responses[v] * coef for v, coef in coefficients.items())
    probability = 1 / (1 + np.exp(-log_odds))

    st.markdown(f"### Estimated Depression Risk: **{probability:.1%}**")

    if probability < 0.25:
        st.success("Low risk")
    elif probability < 0.5:
        st.warning("Moderate risk")
    else:
        st.error("High risk – please consider a professional evaluation.")