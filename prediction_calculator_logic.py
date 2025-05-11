import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
# Import from question_mappings
from question_mappings import QUESTION_MAPPINGS, Z_SCORE_FOR_YES, Z_SCORE_FOR_NO, get_question_by_id

# --- Configuration ---
MODEL_PATH = Path('results/random_forest_model.joblib')
FEATURE_NAMES_PATH = Path('results/model_feature_names.json')

# DEFAULT_SELECTABLE_FEATURES will now be the list of all features defined in QUESTION_MAPPINGS
DEFAULT_SELECTABLE_FEATURES = [q_map['id'] for q_map in QUESTION_MAPPINGS]

# --- Load Model and Feature Names ---
# These are loaded once when the module is imported for efficiency.
RF_MODEL = None
ALL_MODEL_FEATURES = []

try:
    RF_MODEL = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Please ensure the model is trained and saved.")
    # Depending on the application, you might raise an error here or handle it gracefully.
except Exception as e:
    print(f"Error loading model: {e}")

try:
    with open(FEATURE_NAMES_PATH, 'r') as f:
        ALL_MODEL_FEATURES = json.load(f)
    print(f"Feature names loaded successfully from {FEATURE_NAMES_PATH}. Total features: {len(ALL_MODEL_FEATURES)}")
except FileNotFoundError:
    print(f"Error: Feature names file not found at {FEATURE_NAMES_PATH}. Please ensure prepare_rf_data.py has run.")
except Exception as e:
    print(f"Error loading feature names: {e}")

def get_prediction(input_data: dict):
    """
    Predicts depression risk based on user input for selected features.

    Args:
        input_data (dict): A dictionary where keys are feature names/IDs 
                           (e.g., 'cbcl_q86_p') and values are their corresponding
                           'Yes' or 'No' string answers.

    Returns:
        tuple: (predicted_class, prediction_probabilities) or (None, None) if an error occurs.
               predicted_class is an int (0 or 1).
               prediction_probabilities is a list/array of probabilities for [class_0, class_1].
    """
    if RF_MODEL is None or not ALL_MODEL_FEATURES:
        print("Error: Model or feature names not loaded. Cannot make predictions.")
        return None, None

    # Convert Yes/No answers to Z-scores
    z_scored_inputs = {}
    for feature_id, yes_no_answer in input_data.items():
        question_map = get_question_by_id(feature_id)
        if question_map:
            if yes_no_answer == 'Yes':
                z_scored_inputs[feature_id] = question_map['z_score_yes']
            elif yes_no_answer == 'No':
                z_scored_inputs[feature_id] = question_map['z_score_no']
            else:
                print(f"Warning: Invalid answer '{yes_no_answer}' for feature '{feature_id}'. Expected 'Yes' or 'No'. Assuming 0 (mean).")
                # If an invalid answer is passed, it won't be in z_scored_inputs, so it will default to 0 in sample_df
        else:
            print(f"Warning: No mapping found for feature_id '{feature_id}' in QUESTION_MAPPINGS. It will be ignored for Z-score conversion.")


    # Create a DataFrame for the single sample, with all model features, initialized to 0
    # Using 0 as a default because the CBCL features were Z-scored (mean-imputed then Z-scored)
    # so 0 represents the mean for those features. Features not answered by the user (and thus not
    # in z_scored_inputs after Yes/No conversion) will also default to 0.
    sample_df = pd.DataFrame(0, index=[0], columns=ALL_MODEL_FEATURES)

    # Fill in the Z-scored values derived from user's Yes/No answers
    for feature, z_value in z_scored_inputs.items():
        if feature in sample_df.columns:
            sample_df[feature] = z_value # z_value is already a float
        # No warning needed here for unknown features in z_scored_inputs, as it's derived from QUESTION_MAPPINGS

    # Ensure the order of columns in sample_df matches ALL_MODEL_FEATURES (it should by construction)
    # Convert to NumPy array for the model
    sample_np = sample_df[ALL_MODEL_FEATURES].values

    try:
        predicted_class = RF_MODEL.predict(sample_np)[0]
        prediction_probabilities = RF_MODEL.predict_proba(sample_np)[0]
        return int(predicted_class), prediction_probabilities.tolist()
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

# --- Example Usage ---
if __name__ == "__main__":
    print("\n--- Example Usage of Prediction Calculator Logic ---")
    if RF_MODEL is None or not ALL_MODEL_FEATURES:
        print("Cannot run example because model or feature names are not loaded.")
    else:
        # Example: Provide Yes/No answers for a subset of the DEFAULT_SELECTABLE_FEATURES.
        example_yes_no_inputs = {
            DEFAULT_SELECTABLE_FEATURES[0]: 'Yes', # e.g., 'cbcl_q86_p'
            DEFAULT_SELECTABLE_FEATURES[1]: 'No',  # e.g., 'cbcl_q24_p'
            DEFAULT_SELECTABLE_FEATURES[2]: 'Yes', # e.g., 'cbcl_q04_p'
            # Not providing answers for all 20, defaults will be used for the rest
        }
        
        print(f"\nUsing the following {len(example_yes_no_inputs)} Yes/No inputs for prediction example:")
        for f_id, answer in example_yes_no_inputs.items():
            print(f"  - Feature ID {f_id}: {answer}")

        predicted_class, probabilities = get_prediction(example_yes_no_inputs)

        if predicted_class is not None and probabilities is not None:
            print(f"\nExample Prediction:")
            print(f"  Predicted Class: {predicted_class} (0: Likely Lower Risk, 1: Likely Higher Risk - based on 75th percentile)")
            print(f"  Prediction Probabilities: Class 0: {probabilities[0]:.4f}, Class 1: {probabilities[1]:.4f}")
        else:
            print("\nExample Prediction failed.")

        # Example of providing fewer than the 10 selected inputs
        # (the function will use defaults for those not provided)
        print("\n--- Example with a different set of Yes/No inputs ---")
        another_example_inputs = {
            DEFAULT_SELECTABLE_FEATURES[0]: 'No',
            DEFAULT_SELECTABLE_FEATURES[3]: 'Yes', 
            DEFAULT_SELECTABLE_FEATURES[5]: 'No',
        }
        print(f"\nProviding inputs for: {list(another_example_inputs.keys())}")
        predicted_class_partial, probabilities_partial = get_prediction(another_example_inputs)
        if predicted_class_partial is not None and probabilities_partial is not None:
            print(f"\nPartial Input Example Prediction:")
            print(f"  Predicted Class: {predicted_class_partial}")
            print(f"  Prediction Probabilities: Class 0: {probabilities_partial[0]:.4f}, Class 1: {probabilities_partial[1]:.4f}")
        else:
            print("\nPartial Input Example Prediction failed.") 