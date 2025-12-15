import sys
import argparse
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.utils.paths import (
    DICES_BINARY,
    LOGREG_MODEL,
    LOGREG_VECTORIZER,
    SINGLETASK_CHECKPOINT,
    MULTITASK_4_CHECKPOINT,
    LIME_DIR,
    ensure_dir
)

# Import model loading utilities
from models.single_task_transformer import load_finetuned, label2id, id2label
from models.multi_task_transformer import load_model as load_finetuned_multitask

# Import LIME utilities
from explainability.LIME import (
    create_lime_explainer,
    create_classifier,
    explanation_to_rows
)

# LOGISTIC REGRESSION
def run_lime_logreg(
    df,
    logreg_path,
    vectorizer_path,
    output_dir,
    n_samples=5,
    num_features=8,
    seed=67,
    text_col="text",
    label_col="Q_overall_binary",
):
    """
    Run LIME explanations on logistic regression model.
    """
    # Set random seed
    np.random.seed(seed)

    # Load model and vectorizer
    with open(logreg_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Create classifier function for LIME
    classifier_fn = create_classifier(
        model_type='logreg',
        model=model,
        tokenizer=None,
        device=None,
        vectorizer=vectorizer
    )

    # Initialize LIME explainer
    class_names = ["safe", "unsafe"]
    explainer = create_lime_explainer(class_names=class_names)

    # Sample examples to explain
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed)

    # Storage for results
    sample_rows = []
    explanation_rows = []

    # Explain each sample
    for i, row in sample_df.iterrows():
        text = str(row[text_col])
        true_label = row[label_col] if label_col in sample_df.columns else None

        # Get model prediction
        probs = classifier_fn([text])[0]
        pred_id = int(np.argmax(probs))

        # Generate LIME explanation
        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=classifier_fn,
            labels=[1],  # Explain 'unsafe' class
            num_features=num_features,
            num_samples=500,  # Number of perturbed samples for LIME
        )

        # Store sample metadata
        sample_rows.append({
            "model": "logistic_regression",
            "row_index": i,
            "true_label": None if true_label is None else int(true_label),
            "true_label_name": None if true_label is None else class_names[int(true_label)],
            "pred_label": pred_id,
            "pred_label_name": class_names[pred_id],
            "prob_safe": float(probs[0]),
            "prob_unsafe": float(probs[1]),
            "text": text,
        })

        # Store feature attributions
        explanation_rows.extend(
            explanation_to_rows(
                exp,
                row_index=i,
                model_name="logistic_regression",
                label_name="unsafe",
                label_id=1,
                num_features=num_features,
            )
        )

    # Save results
    ensure_dir(output_dir)
    
    samples_df = pd.DataFrame(sample_rows)
    expl_df = pd.DataFrame(explanation_rows)

    samples_path = output_dir / "logreg_lime_samples.csv"
    expl_path = output_dir / "logreg_lime_explanations.csv"
    
    samples_df.to_csv(samples_path, index=False)
    expl_df.to_csv(expl_path, index=False)

# TRANSFORMERS
def run_lime_transformer(
    df,
    model_type,
    checkpoint_path,
    output_dir,
    n_samples=5,
    num_features=5,
    seed=67,
    max_length=256,
    task="Q_overall",
    text_col="text",
    label_col="Q_overall_binary",
):
    """
    Run LIME explanations on single-task or multi-task RoBERTa models.
    """
    # Set random seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Define device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and tokenizer
    if model_type == 'single_task':
        model, tokenizer = load_finetuned(str(checkpoint_path), device=str(device))
        model_name = "single_task"
    elif model_type == 'multi_task':
        model, tokenizer = load_finetuned_multitask(str(checkpoint_path), device=str(device))
        model_name = "multi_task"
    else:
        raise ValueError(f"Invalid model_type: {model_type}. Must be 'single_task' or 'multi_task'")

    # Create classifier function for LIME
    classifier_fn = create_classifier(
        model_type=model_type,
        model=model,
        tokenizer=tokenizer,
        device=device,
        task=task,
        max_length=max_length
    )

    # Initialize LIME explainer
    class_names = ['safe', 'unsafe']
    explainer = create_lime_explainer(class_names=class_names)

    # Sample examples to explain
    sample_df = df.sample(n=min(n_samples, len(df)), random_state=seed)

    # Storage for results
    sample_rows = []
    explanation_rows = []

    # Explain each sample
    for i, row in sample_df.iterrows():
        text = str(row[text_col])
        true_label = row[label_col] if label_col in sample_df.columns else None

        # Get model prediction
        probs = classifier_fn([text])[0]
        pred_id = int(np.argmax(probs))

        # Store sample metadata
        sample_rows.append({
            "model": model_name,
            "row_index": i,
            "true_label": None if true_label is None else int(true_label),
            "true_label_name": None if true_label is None else class_names[int(true_label)],
            "pred_label": pred_id,
            "pred_label_name": class_names[pred_id],
            "prob_safe": float(probs[0]),
            "prob_unsafe": float(probs[1]),
            "text": text,
        })

        # Generate LIME explanation for 'unsafe' class
        unsafe_id = label2id["unsafe"]
        
        exp = explainer.explain_instance(
            text_instance=text,
            classifier_fn=classifier_fn,
            labels=[unsafe_id],
            num_features=num_features,
            num_samples=500,
        )

        # Store feature attributions
        explanation_rows.extend(
            explanation_to_rows(
                exp,
                row_index=i,
                model_name=model_name,
                label_name="unsafe",
                label_id=unsafe_id,
                num_features=num_features,
            )
        )

    # Save results
    ensure_dir(output_dir)
    
    samples_df = pd.DataFrame(sample_rows)
    expl_df = pd.DataFrame(explanation_rows)

    samples_path = output_dir / f"{model_name}_lime_samples.csv"
    expl_path = output_dir / f"{model_name}_lime_explanations.csv"
    
    samples_df.to_csv(samples_path, index=False)
    expl_df.to_csv(expl_path, index=False)

# MAIN
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True, choices=["singletask", "multitask", "logreg"], help="Which model to explain")
    parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint ")
    parser.add_argument("--vectorizer", type=str, help="Path to TF-IDF vectorizer (.pkl)")
    parser.add_argument("--data", type=str, help="Path to dataset CSV")
    parser.add_argument("--output_dir", type=str, help="Output directory for results")
    
    # LIME configuration (all optional with defaults)
    parser.add_argument("--n_samples", type=int, default=5, help="Number of samples to explain")
    parser.add_argument("--num_features", type=int, default=5, help="Number of top features")
    parser.add_argument("--seed", type=int, default=67, help="Random seed")
    parser.add_argument("--max_length", type=int, default=256, help="Max sequence length")
    parser.add_argument("--task", type=str, default="Q_overall", help="Task for multi-task model")
    parser.add_argument("--text_col", type=str, default="text", help="Text column name")
    parser.add_argument("--label_col", type=str, default="Q_overall_binary", help="Label column name")

    args = parser.parse_args()

    # Set default paths
    data_path = Path(args.data) if args.data else DICES_BINARY
    output_dir = Path(args.output_dir) if args.output_dir else LIME_DIR
    
    # Load dataset
    df = pd.read_csv(data_path)

    # Create shared config dict for all models
    config = {
        'n_samples': args.n_samples,
        'num_features': args.num_features,
        'seed': args.seed,
        'text_col': args.text_col,
        'label_col': args.label_col,
    }

    # Run appropriate LIME analysis
    if args.model_type == 'singletask':
        checkpoint = Path(args.checkpoint) if args.checkpoint else SINGLETASK_CHECKPOINT
        run_lime_transformer(
            df=df,
            model_type='single_task',
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            max_length=args.max_length,
            **config
        )
        
    elif args.model_type == 'multitask':
        checkpoint = Path(args.checkpoint) if args.checkpoint else MULTITASK_4_CHECKPOINT
        run_lime_transformer(
            df=df,
            model_type='multi_task',
            checkpoint_path=checkpoint,
            output_dir=output_dir,
            max_length=args.max_length,
            task=args.task,
            **config
        )
        
    elif args.model_type == 'logreg':
        logreg_path = Path(args.checkpoint) if args.checkpoint else LOGREG_MODEL
        vectorizer_path = Path(args.vectorizer) if args.vectorizer else LOGREG_VECTORIZER
        
        # Logistic regression uses more features by default
        logreg_config = config.copy()
        logreg_config['num_features'] = 8 if args.num_features == 5 else args.num_features
        
        run_lime_logreg(
            df=df,
            logreg_path=logreg_path,
            vectorizer_path=vectorizer_path,
            output_dir=output_dir,
            **logreg_config
        )

if __name__ == "__main__":
    main()

'''
Examples:
    python scripts/explainability/run_lime.py --model_type singletask --checkpoint models/checkpoints/best_singletask.pt 
    python scripts/explainability/run_lime.py --model_type multitask --checkpoint models/checkpoints/best_multitask_4.pt 
    python scripts/explainability/run_lime.py --model_type logreg --checkpoint models/checkpoints/logistic_regression_model.pkl
'''