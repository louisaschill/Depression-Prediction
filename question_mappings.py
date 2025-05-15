# question_mappings.py

"""
Mappings from feature IDs to user-facing questions, scales, and Z-score conversions.
This version is updated to use 12 specific questions based on the provided image,
with question text refined for clarity and parent-direction.
"""

QUESTION_MAPPINGS = [
    {
        'id': 'ksads_sleepprob_raw_814_p',
        'question_text': '**Does your child often have trouble sleeping (e.g., difficulty falling or staying asleep)?**',
        'scale_type': 'Binary',
        'options': {'No': 0, 'Yes': 1},
        'z_score_map': {0: -0.75, 1: 0.75}
    },
    {
        'id': 'cbcl_q71_p',
        'question_text': '**Does your child often seem self-conscious or is easily embarrassed?**',
        'scale_type': '3-point',
        'options': {'Not True': 0, 'Somewhat or Sometimes True': 1, 'Very True or Often True': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'cbcl_q04_p',
        'question_text': '**Does your child seem to cry frequently or a lot?**',
        'scale_type': '3-point',
        'options': {'Not True': 0, 'Somewhat or Sometimes True': 1, 'Very True or Often True': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'famhx_ss_parent_prf_p',
        'question_text': '**As the parent completing this, is there a parental history of psychiatric problems (e.g., a diagnosis from a healthcare professional)?**',
        'scale_type': 'Binary',
        'options': {'No': 0, 'Yes': 1},
        'z_score_map': {0: -0.75, 1: 0.75}
    },
    {
        'id': 'sds_p_ss_does',
        'question_text': '**How would you rate your childs typical level of social engagement and actions (e.g., initiating interaction with others)?**',
        'scale_type': '5-point',
        'options': {'1 (Never)': 1, '2': 2, '3 (Sometimes)': 3, '4': 4, '5 (Always)': 5},
        'z_score_map': {1: -1.0, 2: -0.5, 3: 0.0, 4: 0.5, 5: 1.0}
    },
    {
        'id': 'cbcl_q86_p',
        'question_text': '**Does your child often seem to feel overly guilty about things?**',
        'scale_type': '3-point',
        'options': {'Not True': 0, 'Somewhat or Sometimes True': 1, 'Very True or Often True': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'cbcl_q09_p',
        'question_text': '**Does your child often express fears about doing bad things?**',
        'scale_type': '3-point',
        'options': {'Not True': 0, 'Somewhat or Sometimes True': 1, 'Very True or Often True': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'asr_q59_p',
        'question_text': '**Does your child frequently report somatic complaints, such as aches or pains, without a clear medical reason?**',
        'scale_type': '3-point',
        'options': {'Not True': 0, 'Somewhat or Sometimes True': 1, 'Very True or Often True': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'asr_q47_p',
        'question_text': '**Does your child often show symptoms of anxiety, like nervousness or excessive worry?**',
        'scale_type': '3-point',
        'options': {'Not True': 0, 'Somewhat or Sometimes True': 1, 'Very True or Often True': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'cbcl_q112_p',
        'question_text': '**Does your child tend to worry a lot about various things?**',
        'scale_type': '3-point',
        'options': {'Not True': 0, 'Somewhat or Sometimes True': 1, 'Very True or Often True': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    },
    {
        'id': 'sds_p_ss_total',
        'question_text': '**If available, what is your child total score on a social responsiveness scale? (This score typically ranges from 11 to 55).**',
        'scale_type': 'Continuous',
        'options': {'min_value': 11, 'max_value': 55, 'default_value': 33, 'step': 1},
        'z_score_map': "Raw score used"
    },
    {
        'id': 'cbcl_q22_p',
        'question_text': '**Does your child often express feelings of being worthless or inferior?**',
        'scale_type': '3-point',
        'options': {'Not True': 0, 'Somewhat or Sometimes True': 1, 'Very True or Often True': 2},
        'z_score_map': {0: -0.75, 1: 0.0, 2: 0.75}
    }
]

# The get_question_by_id function might not be strictly needed if app.py iterates
# through QUESTION_MAPPINGS directly and prediction_calculator_logic uses a dict.
# If it were needed elsewhere, it would look like this:
# def get_question_by_id(feature_id):
#     """Helper function to retrieve a question mapping by its ID."""
#     for q_map in QUESTION_MAPPINGS:
#         if q_map['id'] == feature_id:
#             return q_map
#     return None