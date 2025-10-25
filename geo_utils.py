import joblib
import numpy as np
import pandas as pd

def predict_geo(csv_file, models_dir="./models"):
    """
    Load gene expression data and predict cancer probability

    Args:
        csv_file: Path to CSV file with gene expression data
        models_dir: Directory containing trained models

    Returns:
        Dictionary with prediction results
    """
    try:
        # Load scaler + model
        scaler = joblib.load(f"{models_dir}/scaler.joblib")
        model = joblib.load(f"{models_dir}/model.joblib")

        with open(f"{models_dir}/threshold.txt") as f:
            threshold = float(f.read().strip())

        # Load gene data
        df = pd.read_csv(csv_file, index_col=0)
        X = scaler.transform(df.values)

        # Predict probability
        proba = model.predict_proba(X)[:, 1].mean()
        result = "Cancer" if proba >= threshold else "Non-cancer"

        return {
            "model_file": f"{models_dir}/model.joblib",
            "probability": float(proba),  # Ensure JSON serializable
            "threshold": float(threshold),
            "decision": result
        }  # Fixed: Added missing closing brace

    except Exception as e:
        return {
            "error": f"Error processing CSV file: {str(e)}",
            "model_file": f"{models_dir}/model.joblib",
            "probability": None,
            "threshold": None,
            "decision": "Error"
        }
