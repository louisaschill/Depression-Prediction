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
        'cbcl_q24_p', 'sleepdisturb4_p', 'cbcl_q76_p', 'cbcl_q102_p',
        'cbcl_q01_p', 'cbcl_q61_p', 'famhx_ss_parent_prf_p',
        'kbi_p_conflict', 'ksads_asd_raw_560_p', 'asr_q120_p',
        'asr_q116_p', 'sai_ss_basket_nyr_p'
    ]
    return df[top_vars].dropna()

baseline_df = load_baseline()

# ----------------------------------------
# 3. Logistic Regression Coefficients
# ----------------------------------------
intercept = -0.5833

coefficients = {
    'cbcl_q24_p': 0.2902,
    'sleepdisturb4_p': 0.3003,
    'cbcl_q76_p': 0.1844,
    'cbcl_q102_p': 0.2228,
    'cbcl_q01_p': 0.1467,
    'cbcl_q61_p': 0.0356,
    'famhx_ss_parent_prf_p': 0.1889,
    'kbi_p_conflict': 0.1825,
    'ksads_asd_raw_560_p': 0.1438,
    'asr_q120_p': 0.1244,
    'asr_q116_p': 0.1827,
    'sai_ss_basket_nyr_p': -0.1044
}

labels = {
    'cbcl_q24_p': "Child has poor appetite",
    'sleepdisturb4_p': "Difficulty falling asleep",
    'cbcl_q76_p': "Sleeps less than other children",
    'cbcl_q102_p': "Lacks energy / slow moving",
    'cbcl_q01_p': "Acts too young for age",
    'cbcl_q61_p': "Poor school work",
    'famhx_ss_parent_prf_p': "Parent has psychiatric condition",
    'kbi_p_conflict': "Family conflict level",
    'ksads_asd_raw_560_p': "Repetitive behavior (ASD screener)",
    'asr_q120_p': "Drives too fast (parent self-report)",
    'asr_q116_p': "Gets upset too easily (parent self-report)",
    'sai_ss_basket_nyr_p': "Played basketball in past year"
}

# ----------------------------------------
# 4. Inputs + Plots
# ----------------------------------------
responses = {}
for var in coefficients:
    label = labels.get(var, var)
    col1, col2 = st.columns([2, 3])

    with col1:
        if var in ['famhx_ss_parent_prf_p', 'ksads_asd_raw_560_p', 'sai_ss_basket_nyr_p']:
            val = st.selectbox(f"{label} (0 = No, 1 = Yes)", options=[0, 1], index=0, key=var)
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