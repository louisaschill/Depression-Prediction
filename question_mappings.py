# question_mappings.py

# This file stores the mapping between user-friendly questions and model features,
# including how Yes/No answers translate to Z-scores for the model.

# Z-score assignments for Yes/No answers (simplified initial approach)
Z_SCORE_FOR_YES = 0.75
Z_SCORE_FOR_NO = -0.75

# Top 20 features from the Random Forest model that will be used for the questionnaire
# For actual CBCL items, the question_text should be the specific item wording.
# The current question_text entries are generic placeholders.
QUESTION_MAPPINGS = [
    {'id': 'cbcl_q86_p', 'question_text': 'Regarding CBCL item Q86: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q24_p', 'question_text': 'Regarding CBCL item Q24: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q04_p', 'question_text': 'Regarding CBCL item Q04: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q100_p', 'question_text': 'Regarding CBCL item Q100: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q71_p', 'question_text': 'Regarding CBCL item Q71: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q44_p', 'question_text': 'Regarding CBCL item Q44: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q112_p', 'question_text': 'Regarding CBCL item Q112: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q103_p', 'question_text': 'Regarding CBCL item Q103: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q88_p', 'question_text': 'Regarding CBCL item Q88: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q03_p', 'question_text': 'Regarding CBCL item Q03: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q12_p', 'question_text': 'Regarding CBCL item Q12: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q78_p', 'question_text': 'Regarding CBCL item Q78: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q32_p', 'question_text': 'Regarding CBCL item Q32: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q87_p', 'question_text': 'Regarding CBCL item Q87: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q08_p', 'question_text': 'Regarding CBCL item Q08: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q50_p', 'question_text': 'Regarding CBCL item Q50: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q56f_p', 'question_text': 'Regarding CBCL item Q56F: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q09_p', 'question_text': 'Regarding CBCL item Q09: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q56b_p', 'question_text': 'Regarding CBCL item Q56B: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
    {'id': 'cbcl_q35_p', 'question_text': 'Regarding CBCL item Q35: Does the child often exhibit this behavior/trait?', 'z_score_yes': Z_SCORE_FOR_YES, 'z_score_no': Z_SCORE_FOR_NO},
]

def get_question_by_id(feature_id):
    """Helper function to retrieve a question mapping by its ID."""
    for q_map in QUESTION_MAPPINGS:
        if q_map['id'] == feature_id:
            return q_map
    return None 