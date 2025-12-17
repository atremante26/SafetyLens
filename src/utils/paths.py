from pathlib import Path

# Project root (3 levels up from this file: src/utils/paths.py)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# DATA PATHS
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# Main dataset
DICES_RAW = RAW_DATA_DIR / 'diverse_safety_adversarial_dialog_350.csv'
DICES_BINARY = PROCESSED_DATA_DIR / 'dices_350_binary.csv'

# MODEL PATHS 
MODELS_DIR = PROJECT_ROOT / 'models'
CHECKPOINTS_DIR = MODELS_DIR / 'checkpoints'

# Model checkpoints
LOGREG_MODEL = CHECKPOINTS_DIR / 'logistic_regression_model.pkl'
LOGREG_VECTORIZER = CHECKPOINTS_DIR / 'tfidf_vectorizer.pkl'
SINGLETASK_CHECKPOINT = CHECKPOINTS_DIR / 'best_singletask.pt'
MULTITASK_2_CHECKPOINT = CHECKPOINTS_DIR / 'best_multitask_2.pt'
MULTITASK_4_CHECKPOINT = CHECKPOINTS_DIR / 'best_multitask_4.pt'

# RESULTS PATHS
RESULTS_DIR = PROJECT_ROOT / 'results'

# Model predictions
PREDICTIONS_DIR = RESULTS_DIR / 'predictions' 
LOGREG_PREDS_DIR = PREDICTIONS_DIR / 'logistic_regression'
SINGLETASK_PREDS_DIR = PREDICTIONS_DIR / 'single_task_transformer'
MULTITASK_PREDS_DIR = PREDICTIONS_DIR / 'multi_task_transformer'

# Evaluation metrics
EVALUATION_DIR = RESULTS_DIR / 'evaluation'

# Explainability outputs
EXPLAINABILITY_DIR = RESULTS_DIR / 'explainability'
LIME_DIR = EXPLAINABILITY_DIR / 'lime'
SHAP_DIR = EXPLAINABILITY_DIR / 'shap'
IG_DIR = EXPLAINABILITY_DIR / 'ig'

# Figures
FIGURES_DIR = RESULTS_DIR / 'figures'

# HELPER FUNCTIONS
def ensure_dir(path):
    path = Path(path)
    # If path has a file extension, get the parent directory
    if path.suffix: 
        directory = path.parent
    else:
        directory = path
    
    # Create directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)

def get_prediction_path(model_name, filename):
    """Get prediction file path for a given model."""
    model_dirs = {
        'logreg': LOGREG_PREDS_DIR,
        'logistic_regression': LOGREG_PREDS_DIR,
        'single_task': SINGLETASK_PREDS_DIR,
        'singletask': SINGLETASK_PREDS_DIR,
        'multi_task_2': MULTITASK_PREDS_DIR,
        'multi_task_4': MULTITASK_PREDS_DIR,
        'multitask_2': MULTITASK_PREDS_DIR,
        'multitask_4': MULTITASK_PREDS_DIR,
    }
    
    model_dir = model_dirs.get(model_name.lower())
    if model_dir is None:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_dir / filename