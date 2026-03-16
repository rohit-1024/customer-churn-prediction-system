


from pathlib import Path
import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "saved_models" / "best_churn_prediction_model.pkl"
COLUMNS_PATH = PROJECT_ROOT / "saved_models" / "training_columns.pkl"


def load_model():

    model = joblib.load(MODEL_PATH)

    training_columns = joblib.load(COLUMNS_PATH)

    return model, training_columns
