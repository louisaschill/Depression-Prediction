import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
# Import from question_mappings - ONLY import QUESTION_MAPPINGS
from question_mappings import QUESTION_MAPPINGS #, Z_SCORE_FOR_YES, Z_SCORE_FOR_NO, get_question_by_id # REMOVED UNUSED IMPORTS

# --- Configuration ---
MODEL_PATH = os.path.join("results", "random_forest_model.joblib")
FEATURE_NAMES_PATH = os.path.join("results", "model_feature_names.json")

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

# Create a lookup dictionary for Z-score maps for faster access
z_score_lookup = {item['id']: item['z_score_map'] for item in QUESTION_MAPPINGS if 'id' in item and 'z_score_map' in item}

def get_prediction(input_data: dict):
    """
    Generates a prediction based on user input from the questionnaire.

    Args:
        input_data (dict): A dictionary where keys are feature IDs (e.g., 'cbcl_q86_p')
                           and values are the user's selected options 
                           (e.g., 0, 1, 2 for '012' scale; 'Yes', 'No' for 'YN' scale).

    Returns:
        tuple: (predicted_class, prediction_probabilities) or (None, None) if model not loaded.
               predicted_class is 0 (Low Risk) or 1 (Higher Risk).
               prediction_probabilities is a list like [prob_class_0, prob_class_1].
    """
    if RF_MODEL is None or not ALL_MODEL_FEATURES:
        print("Model or feature names not loaded. Cannot make prediction.")
        return None, None

    # Create the full feature vector, ordered according to model_feature_names
    # Default to Z-score 0 (mean) for any feature
    input_vector_dict = {feature: 0.0 for feature in ALL_MODEL_FEATURES}

    # Populate the vector with Z-scores based on user input
    for feature_id, selected_option in input_data.items():
        if feature_id in ALL_MODEL_FEATURES:
            if feature_id in z_score_lookup:
                # Find the correct Z-score using the selected option as the key
                z_score_map = z_score_lookup[feature_id]
                # Important: Ensure selected_option type matches keys in z_score_map 
                # (e.g., user input might be int 0, map key might be int 0)
                # The keys in input_data values should directly map to keys in z_score_map
                if selected_option in z_score_map:
                    input_vector_dict[feature_id] = z_score_map[selected_option]
                else:
                    print(f"Warning: Selected option '{selected_option}' for feature '{feature_id}' not found in z_score_map. Defaulting to 0.")
                    # Keep the default 0.0 assigned earlier
            else:
                # This case should ideally not happen if QUESTION_MAPPINGS covers all input features
                print(f"Warning: Z-score map not found for feature '{feature_id}'. Defaulting to 0.")
                # Keep the default 0.0 assigned earlier
        # else:
            # Feature ID from input_data is not in the model's expected features. Ignore it.
            # print(f"Warning: Input feature '{feature_id}' not recognized by the model. Ignoring.")

    # Convert the dictionary to a list in the correct order
    input_vector = [input_vector_dict[feature] for feature in ALL_MODEL_FEATURES]

    # Reshape for the model (expects a 2D array)
    input_array = np.array(input_vector).reshape(1, -1)

    # Make prediction
    prediction = RF_MODEL.predict(input_array)
    probabilities = RF_MODEL.predict_proba(input_array)

    predicted_class = int(prediction[0])
    prediction_probabilities = probabilities[0].tolist() # Convert to list [prob_0, prob_1]

    return predicted_class, prediction_probabilities

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

        # Example simulating user input from the questionnaire
        # Note: Keys in example_input should match the *values* passed from Streamlit radio buttons
        # which correspond to the *keys* in the 'options' dict in question_mappings.py
        example_input = {
            'cbcl_q86_p': 2,      # User selected 'Often' (value 2)
            'cbcl_q24_p': 0,      # User selected 'Never' (value 0)
            'cbcl_q04_p': 1,      # User selected 'Sometimes' (value 1)
            'cbcl_q100_p': 1,     
            'cbcl_q71_p': 0,
            'cbcl_q44_p': 'No',   # User selected 'No' (value 'No')
            'cbcl_q112_p': 2,
            'cbcl_q103_p': 1,
            'cbcl_q88_p': 0,
            'cbcl_q03_p': 1,
            'cbcl_q66_p': 0,
            'cbcl_q31_p': 'Yes',  # User selected 'Yes' (value 'Yes')
            'cbcl_q90_p': 2,
            'cbcl_q65_p': 0,
            'cbcl_q11_p': 1,
            'cbcl_q75_p': 'No',
            'cbcl_q45_p': 1,
            'cbcl_q104_p': 0,
            'cbcl_q13_p': 1,
            'cbcl_q29_p': 'No'
            # Assuming user answered all 20 questions
        }

        print("\nExample Input (Simulated User Answers):")
        print(example_input)
        
        pred_class, pred_probs = get_prediction(example_input)

        if pred_class is not None:
            print(f"\nPredicted Class: {pred_class} ({'Higher Risk' if pred_class == 1 else 'Low Risk'})")
            print(f"Prediction Probabilities: [P(Low Risk), P(Higher Risk)] = {pred_probs}")

        # Example with missing input (should default to Z-score 0)
        example_missing_input = {
            'cbcl_q103_p': 2, # Unhappy, sad, depressed = Often
            'cbcl_q11_p': 2,  # Cries a lot = Often
        }
        print("\nExample with Missing Input:")
        print(example_missing_input)
        pred_class_missing, pred_probs_missing = get_prediction(example_missing_input)
        if pred_class_missing is not None:
            print(f"\nPredicted Class (Missing Input): {pred_class_missing} ({'Higher Risk' if pred_class_missing == 1 else 'Low Risk'})")
            print(f"Prediction Probabilities (Missing Input): {pred_probs_missing}") 