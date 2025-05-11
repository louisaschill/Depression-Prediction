import joblib
import pandas as pd
import numpy as np
import json
from pathlib import Path

# --- Configuration ---
MODEL_PATH = Path('results/random_forest_model.joblib')
FEATURE_NAMES_PATH = Path('results/model_feature_names.json')

# Top 10 features identified from Random Forest feature importances
# (as an example, a real application might make this configurable)
SELECTED_INPUT_FEATURES = [
    'cbcl_q86_p', 'cbcl_q24_p', 'cbcl_q04_p', 'cbcl_q100_p', 'cbcl_q71_p',
    'cbcl_q44_p', 'cbcl_q112_p', 'cbcl_q103_p', 'cbcl_q88_p', 'cbcl_q03_p'
]

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
        input_data (dict): A dictionary where keys are feature names 
                           (subset of SELECTED_INPUT_FEATURES or all of them)
                           and values are their corresponding Z-scored numerical inputs.

    Returns:
        tuple: (predicted_class, prediction_probabilities) or (None, None) if an error occurs.
               predicted_class is an int (0 or 1).
               prediction_probabilities is a list/array of probabilities for [class_0, class_1].
    """
    if RF_MODEL is None or not ALL_MODEL_FEATURES:
        print("Error: Model or feature names not loaded. Cannot make predictions.")
        return None, None

    # Create a DataFrame for the single sample, with all model features, initialized to 0
    # Using 0 as a default because the CBCL features were Z-scored (mean-imputed then Z-scored)
    # so 0 represents the mean for those features.
    # For any one-hot encoded features (if they were part of ALL_MODEL_FEATURES and not CBCL),
    # 0 is also a safe default, meaning that category is not present.
    sample_df = pd.DataFrame(0, index=[0], columns=ALL_MODEL_FEATURES)

    # Fill in the values provided by the user
    # We assume input_data keys are valid feature names and values are already preprocessed (Z-scored)
    for feature, value in input_data.items():
        if feature in sample_df.columns:
            try:
                sample_df[feature] = float(value) # Ensure value is float for safety
            except ValueError:
                print(f"Warning: Could not convert value '{value}' for feature '{feature}' to float. Using 0.")
                # Keep default of 0 if conversion fails
        else:
            # This should ideally not happen if input_data only contains known features
            print(f"Warning: Unknown feature '{feature}' in input_data. It will be ignored.")

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
        # Example: Provide Z-scored inputs for the top 10 selected features.
        # For a real application, these values would come from a user interface.
        # Using placeholder Z-score values (e.g., average, or one std dev above/below mean)
        example_inputs = {
            'cbcl_q86_p': 0.5,   # Example: 0.5 standard deviations above the mean
            'cbcl_q24_p': -0.2,  # Example: 0.2 standard deviations below the mean
            'cbcl_q04_p': 0.0,   # Example: at the mean
            'cbcl_q100_p': 1.0,  
            'cbcl_q71_p': -1.5, 
            'cbcl_q44_p': 0.3,  
            'cbcl_q112_p': 0.0,  
            'cbcl_q103_p': -0.1, 
            'cbcl_q88_p': 2.0,  
            'cbcl_q03_p': 0.7   
        }
        
        print(f"\nUsing the following {len(SELECTED_INPUT_FEATURES)} features for prediction example:")
        for f in SELECTED_INPUT_FEATURES:
            print(f"  - {f}: {example_inputs.get(f, '(default will be used if not in example_inputs)')}")

        # Ensure our example inputs only contain the defined SELECTED_INPUT_FEATURES for this basic example
        valid_example_inputs = {k: v for k, v in example_inputs.items() if k in SELECTED_INPUT_FEATURES}

        predicted_class, probabilities = get_prediction(valid_example_inputs)

        if predicted_class is not None and probabilities is not None:
            print(f"\nExample Prediction:")
            print(f"  Predicted Class: {predicted_class} (0: Likely Lower Risk, 1: Likely Higher Risk - based on 75th percentile)")
            print(f"  Prediction Probabilities: Class 0: {probabilities[0]:.4f}, Class 1: {probabilities[1]:.4f}")
        else:
            print("\nExample Prediction failed.")

        # Example of providing fewer than the 10 selected inputs
        # (the function will use defaults for those not provided)
        print("\n--- Example with fewer inputs (defaults used for missing) ---")
        partial_example_inputs = {
            'cbcl_q86_p': 1.5,   
            'cbcl_q04_p': -0.5,
            'cbcl_q100_p': 0.8
        }
        print(f"\nProviding inputs for: {list(partial_example_inputs.keys())}")
        predicted_class_partial, probabilities_partial = get_prediction(partial_example_inputs)
        if predicted_class_partial is not None and probabilities_partial is not None:
            print(f"\nPartial Input Example Prediction:")
            print(f"  Predicted Class: {predicted_class_partial}")
            print(f"  Prediction Probabilities: Class 0: {probabilities_partial[0]:.4f}, Class 1: {probabilities_partial[1]:.4f}")
        else:
            print("\nPartial Input Example Prediction failed.") 