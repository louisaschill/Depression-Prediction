# question_mappings.py

"""
Mappings from feature IDs to user-facing questions, scales, and Z-score conversions.
This version is updated to use 12 specific questions.
"""

QUESTION_MAPPINGS = [
    {
        'id': 'cbcl_q24_p',
        'question_text': '**How often does your child have a poor appetite?**',
        'scale_type': '012',
        'options': {'Never': 0, 'Sometimes': 1, 'Often': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'sleepdisturb4_p',
        'question_text': '**How often does your child have difficulty falling asleep?**',
        'scale_type': '012',
        'options': {'Rarely/Never': 0, 'Sometimes': 1, 'Often/Always': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'cbcl_q76_p',
        'question_text': '**Does your child generally sleep less than other children their age?**',
        'scale_type': 'YN',
        'options': {'Yes': 'Yes', 'No': 'No'},
        'z_score_map': {'Yes': 0.75, 'No': -0.75} # Assuming sleeping less is a risk factor
    },
    {
        'id': 'cbcl_q102_p',
        'question_text': '**How often does your child seem to lack energy or appear slow-moving?**',
        'scale_type': '012',
        'options': {'Never': 0, 'Sometimes': 1, 'Often': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'cbcl_q01_p',
        'question_text': '**How often does your child act much younger than their actual age?**',
        'scale_type': '012',
        'options': {'Never': 0, 'Sometimes': 1, 'Often': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'cbcl_q61_p', # cbcl_q66_p was "Poor school work"
        'question_text': '**How often is your child\'s school work considered poor?**',
        'scale_type': '012',
        'options': {'Never': 0, 'Sometimes': 1, 'Often': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'famhx_ss_parent_prf_p',
        'question_text': '**As the parent completing this, have you been diagnosed with a psychiatric condition by a professional?**',
        'scale_type': 'YN',
        'options': {'Yes': 'Yes', 'No': 'No'},
        'z_score_map': {'Yes': 0.75, 'No': -0.75} # Assuming parent condition is a risk factor for child
    },
    {
        'id': 'kbi_p_conflict',
        'question_text': '**How would you rate the typical level of conflict within your immediate family?**',
        'scale_type': '012', # Mapping to Low/Medium/High
        'options': {'Low': 0, 'Medium': 1, 'High': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75} # Assuming High conflict is riskier
    },
    {
        'id': 'ksads_asd_raw_560_p',
        'question_text': '**Does your child often engage in repetitive behaviors (e.g., movements, routines, or speech)?**',
        'scale_type': 'YN',
        'options': {'Yes': 'Yes', 'No': 'No'},
        'z_score_map': {'Yes': 0.75, 'No': -0.75} # Assuming repetitive behavior is a risk factor
    },
    {
        'id': 'asr_q120_p',
        'question_text': '**As a parent, do you often find yourself driving significantly faster than the speed limit or conditions allow?**',
        'scale_type': 'YN',
        'options': {'Yes': 'Yes', 'No': 'No'},
        'z_score_map': {'Yes': 0.5, 'No': -0.5} # Assuming this parent behavior might be a moderate risk factor
    },
    {
        'id': 'asr_q116_p',
        'question_text': '**As a parent, do you find yourself getting upset more easily or intensely than you would like?**',
        'scale_type': 'YN',
        'options': {'Yes': 'Yes', 'No': 'No'},
        'z_score_map': {'Yes': 0.75, 'No': -0.75} # Assuming this parent trait is a risk factor
    },
    {
        'id': 'sai_ss_basket_nyr_p',
        'question_text': '**Has your child participated in playing basketball in the past year?**',
        'scale_type': 'YN',
        'options': {'Yes': 'Yes', 'No': 'No'},
        'z_score_map': {'Yes': -0.25, 'No': 0.25} # Assuming Yes (activity) is slightly protective
    }
]

# The get_question_by_id function is no longer needed by prediction_calculator_logic.py
# as it now uses a direct lookup from a dictionary generated from QUESTION_MAPPINGS.
# If it were needed elsewhere, it would look like this:
# def get_question_by_id(feature_id):
#     """Helper function to retrieve a question mapping by its ID."""
#     for q_map in QUESTION_MAPPINGS:
#         if q_map['id'] == feature_id:
#             return q_map
#     return None