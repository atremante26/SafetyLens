import sys
from pathlib import Path
import argparse
import random

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

from captum.attr import IntegratedGradients
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.paths import DICES_BINARY, IG_DIR, ensure_dir
from models.single_task_transformer import load_finetuned
from models.multi_task_transformer import load_model as load_finetuned_multitask

# Configuration
SEED = 67
BASELINE_TOKEN = "[PAD]"  # Baseline for IG


# UTILS
def set_seed(seed=67):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_predictions(model, tokenizer, texts, device, task=None, batch_size=32):
    """
    Get model predictions for selecting samples by confidence.
    Uses batching to avoid memory issues on large datasets.
    """
    model.eval()
    all_probs = []
    
    # Process in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Getting predictions", leave=False):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize batch
        encoding = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        with torch.no_grad():
            if task:  # Multi-task
                outputs = model(input_ids, attention_mask)
                logits = outputs[task].squeeze(-1)
                probs = torch.sigmoid(logits)
            else:  # Single-task
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1]  # P(unsafe)
        
        all_probs.extend(probs.cpu().numpy().tolist())
    
    return np.array(all_probs)


def select_samples_by_confidence(df, model, tokenizer, device, n_samples, confidence_split, task=None):
    """
    Select samples based on prediction confidence.
    """
    # Get predictions for entire test set
    texts = df['text'].tolist()
    probs = get_predictions(model, tokenizer, texts, device, task)
    df = df.copy()
    df['pred_prob'] = probs
    
    if confidence_split == 'all':
        # Random sample
        selected = df.sample(n=min(n_samples, len(df)), random_state=SEED)
        confidence_labels = ['all'] * len(selected)
    
    elif confidence_split == 'high':
        # High-confidence unsafe (prob > 0.8)
        high_conf = df[df['pred_prob'] > 0.8].nlargest(n_samples, 'pred_prob')
        selected = high_conf
        confidence_labels = ['high'] * len(selected)
    
    elif confidence_split == 'borderline':
        # Borderline (0.45 <= prob <= 0.55)
        df['distance_to_threshold'] = abs(df['pred_prob'] - 0.5)
        borderline = df[
            (df['pred_prob'] >= 0.45) & (df['pred_prob'] <= 0.55)
        ].nsmallest(n_samples, 'distance_to_threshold')
        selected = borderline
        confidence_labels = ['borderline'] * len(selected)
    
    elif confidence_split == 'high_vs_borderline':
        # Half high-confidence, half borderline
        n_each = n_samples // 2
        
        # High-confidence
        high_conf = df[df['pred_prob'] > 0.8].nlargest(n_each, 'pred_prob')
        
        # Borderline
        df['distance_to_threshold'] = abs(df['pred_prob'] - 0.5)
        borderline = df[
            (df['pred_prob'] >= 0.45) & (df['pred_prob'] <= 0.55)
        ].nsmallest(n_each, 'distance_to_threshold')
        
        # Combine
        selected = pd.concat([high_conf, borderline])
        confidence_labels = ['high'] * len(high_conf) + ['borderline'] * len(borderline)
    
    else:
        raise ValueError(f"Unknown confidence_split: {confidence_split}")
    
    return selected.reset_index(drop=True), confidence_labels


# INTEGRATED GRADIENTS
def compute_ig_attributions(model, tokenizer, text, device, task=None, n_steps=50):
    """
    Compute Integrated Gradients attributions for a single text.
    """
    model.eval()
    
    # Tokenize
    encoding = tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get input embeddings
    if task:  # Multi-task
        embeddings = model.roberta.embeddings.word_embeddings(input_ids)
    else:  # Single-task
        embeddings = model.roberta.embeddings.word_embeddings(input_ids)
    
    # Create baseline (zero embeddings)
    baseline_embeddings = torch.zeros_like(embeddings)
    
    def forward_func(embeddings, attn_mask):
        outputs = model.roberta(inputs_embeds=embeddings, attention_mask=attn_mask)
        seq = outputs.last_hidden_state 

        if task:  # multitask
            cls = seq[:, 0, :]
            logits = model.heads[task](model.dropout(cls)).squeeze(-1)  
            return torch.sigmoid(logits).unsqueeze(-1) 

        else:  # single-task HF RobertaForSequenceClassification
            logits = model.classifier(seq) 
            if logits.shape[-1] == 2:
                probs = torch.softmax(logits, dim=-1)[:, 1:2]
            else:
                probs = torch.sigmoid(logits)  
            return probs


    # Initialize IG
    ig = IntegratedGradients(forward_func)
    
    with torch.no_grad():
        test_out = forward_func(embeddings, attention_mask)

    # Compute attributions on embeddings
    attributions, delta = ig.attribute(
        inputs=embeddings,
        baselines=baseline_embeddings,
        additional_forward_args=(attention_mask,),
        n_steps=n_steps,
        return_convergence_delta=True
    )
    
    # Sum attributions across embedding dimensions to get per-token attribution
    token_attributions = (
        attributions
        .sum(dim=-1)
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
    )

    
    # Get prediction
    with torch.no_grad():
        pred_prob = forward_func(embeddings, attention_mask).item()

    # Convert to tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    return tokens, token_attributions, pred_prob, delta.item()


def run_ig_analysis(
    df,
    model,
    tokenizer,
    device,
    model_name,
    task=None,
    n_steps=50,
    confidence_labels=None
):
    """
    Run IG analysis on all samples in dataframe.
    """
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing IG"):
        text = row['text']
        true_label = row.get('Q_overall_binary', None)
        confidence_level = confidence_labels[idx] if confidence_labels else 'all'
        
        # Compute IG
        tokens, attributions, pred_prob, convergence_delta = compute_ig_attributions(
            model, tokenizer, text, device, task, n_steps
        )
        
        # Store results for each token
        for token, attr in zip(tokens, attributions):
            # Skip special tokens in output
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>']:
                continue
            
            results.append({
                'model': model_name,
                'task': task if task else 'Q_overall',
                'example_idx': idx,
                'confidence_level': confidence_level,
                'true_label': true_label,
                'pred_prob': pred_prob,
                'convergence_delta': convergence_delta,
                'token': token,
                'attribution': attr,
                'abs_attribution': abs(attr),
            })
    
    return pd.DataFrame(results)


# MAIN
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to model checkpoint")
    parser.add_argument('--model_type', type=str, required=True, choices=['singletask', 'multitask'])
    parser.add_argument('--task', type=str, default='Q_overall', help="Task name (for multi-task)")
    parser.add_argument('--data', type=str, help="Path to dataset CSV")
    parser.add_argument('--output_dir', type=str, help="Output directory")
    
    # Sampling configuration
    parser.add_argument('--n_samples', type=int, default=30, help="Number of samples to explain")
    parser.add_argument(
        '--confidence_split',
        type=str,
        default='all',
        choices=['all', 'high', 'borderline', 'high_vs_borderline'],
        help="How to select samples by confidence"
    )
    
    # IG configuration
    parser.add_argument('--n_steps', type=int, default=50, help="Number of IG steps")
    parser.add_argument('--seed', type=int, default=67, help="Random seed")
    
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    data_path = Path(args.data) if args.data else DICES_BINARY
    output_dir = Path(args.output_dir) if args.output_dir else IG_DIR
    ensure_dir(output_dir)
    
    # Load data
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} samples")
    
    # Load model
    print(f"\nLoading {args.model_type} model...")
    if args.model_type == 'singletask':
        model, tokenizer = load_finetuned(args.checkpoint, device=str(device))
        model_name = 'single_task'
        task = None
    else:
        model, tokenizer = load_finetuned_multitask(args.checkpoint, device=str(device))
        model_name = 'multi_task'
        task = args.task
    
    # Select samples by confidence
    selected_df, confidence_labels = select_samples_by_confidence(
        df, model, tokenizer, device, args.n_samples, args.confidence_split, task
    )
    
    # Run IG analysis
    print(f"\nRunning Integrated Gradients (n_steps={args.n_steps})...")
    results_df = run_ig_analysis(
        selected_df,
        model,
        tokenizer,
        device,
        model_name,
        task=task,
        n_steps=args.n_steps,
        confidence_labels=confidence_labels
    )
    
    # Save results
    results_df.to_csv(output_dir, index=False)
    print(f"\nSaved IG results: {output_dir}")


if __name__ == "__main__":
    main()
"""
Usage:
NORMAL:
python scripts/explainability/run_integrated_gradients.py \
    --checkpoint models/checkpoints/best_singletask.pt \
    --model_type singletask \
    --task Q_overall \
    --n_samples 10
    --output_dir results/explainability/ig/ig_single_q.csv

#EXPERIMENT 3:
# Single-task on Q_overall
python scripts/explainability/run_integrated_gradients.py \
    --checkpoint models/checkpoints/best_singletask.pt \
    --model_type singletask \
    --task Q_overall \
    --n_samples 30 \
    --confidence_split high_vs_borderline \
    --n_steps 50 \
    --output_dir results/evaluation/experiment3/experiment3_single.csv

# Multi-task-4 on Q_overall  
python scripts/explainability/run_integrated_gradients.py \
    --checkpoint models/checkpoints/best_multitask_4.pt \
    --model_type multitask \
    --task Q_overall \
    --n_samples 30 \
    --confidence_split high_vs_borderline \
    --n_steps 50 \
    --output_dir results/evaluation/experiment3/experiment3_multi4.csv
"""