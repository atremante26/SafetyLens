import sys
from pathlib import Path
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.multi_task_transformer import MultiTaskRoBERTa, MultiTaskDataset
from src.data_preprocessing import load_multi_task_data

# HYPERPARAMETERS 
SEED = 42
MODEL_NAME = "roberta-base"
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3          # set 5 manually when running 4-head 
LR = 2e-5
WARMUP_FRAC = 0.1
NUM_WORKERS = 0     
BALANCE = False     # multi-task: keep False, use pos_weight instead

# Task configurations
TASKS_2 = ["Q_overall", "Q2_harmful"]
TASKS_4 = ["Q_overall", "Q2_harmful", "Q3_bias", "Q6_policy"]

# Mapping from task names to dataset column names
TASK_TO_COL = {
    "Q_overall": "Q_overall_binary",
    "Q2_harmful": "Q2_harmful_binary",
    "Q3_bias": "Q3_bias_binary",
    "Q6_policy": "Q6_policy_binary",
}

# UTILS
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    '''
    Compute comprehensive binary classification metrics
    '''
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_pos": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_pos": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_pos": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0,
        "pred_pos_rate": float(y_pred.mean()),
        "true_pos_rate": float(y_true.mean()),
    }

def compute_pos_weights(train_df: pd.DataFrame, tasks: List[str]):
    '''
    Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance.

    pos_weight = (# negative examples) / (# positive examples)
    
    Higher pos_weight increases penalty for misclassifying the minority class,
    helping the model learn from imbalanced data without undersampling.
    '''
    out = {}
    for t in tasks:
        col = TASK_TO_COL[t]
        y = train_df[col].astype(int).values
        pos = int((y == 1).sum())
        neg = int((y == 0).sum())
        out[t] = float(neg / max(pos, 1)) # Avoid division by 0
    return out


# TRAIN / EVAL / PRED
def train_epoch(model, loader, tasks, device, loss_fns, optimizer, scheduler, epoch_idx):
    '''
    Train model for one epoch across all tasks.
    
    For each batch:
    1. Forward pass through shared encoder
    2. Compute loss for each task independently
    3. Average losses across tasks
    4. Backpropagate and update weights
    '''
    model.train()
    running = 0.0

    for batch in tqdm(loader, desc=f"Train epoch {epoch_idx+1}", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask)

        # Compute loss for each task and sum
        loss = 0.0
        for t in tasks:
            y = batch[f"labels_{t}"].to(device).float()
            logits = outputs[t]  # [B]
            loss = loss + loss_fns[t](logits, y)

        # Average loss across tasks for gradient updates
        loss = loss / len(tasks)

        # Backpropagation with gradient clipping
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Prevent exploding gradients
        optimizer.step()
        scheduler.step() # Update learning rate 

        running += float(loss.item())

    return running / max(len(loader), 1)

@torch.no_grad()
def evaluate(model, loader, tasks, device, loss_fns):
    '''
    Evaluate model on validation or test set
    '''
    model.eval()
    running = 0.0

    # Collect predictions and ground truth for each task
    all_true = {t: [] for t in tasks}
    all_prob = {t: [] for t in tasks}

    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)

        loss = 0.0
        for t in tasks:
            y = batch[f"labels_{t}"].to(device).float()
            logits = outputs[t]
            loss = loss + loss_fns[t](logits, y)

            # Convert logits to probabilities and collect predictions
            prob = torch.sigmoid(logits).squeeze(-1).detach().cpu().numpy()   
            true = y.squeeze(-1).detach().cpu().numpy()                     
            all_prob[t].extend(prob.tolist())
            all_true[t].extend(true.tolist())

        loss = loss / len(tasks)
        running += float(loss.item())

    # Compute metrics for each task
    metrics = {t: compute_binary_metrics(all_true[t], all_prob[t]) for t in tasks}
    return running / max(len(loader), 1), metrics

@torch.no_grad()
def predict(model, loader, df_source, tasks, device):
    '''
    Generate predictions on test set for all tasks.
    
    Creates a dataframe with:
    - Original text
    - Ground truth labels (task_true)
    - Predicted probabilities (task_prob)
    '''
    model.eval()
    rows = []
    df_source = df_source.reset_index(drop=True)
    idx = 0

    for batch in tqdm(loader, desc="Predict", leave=False):
        bs = batch["input_ids"].shape[0]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        outputs = model(input_ids, attention_mask)

        # Build prediction row for each example in batch
        for i in range(bs):
            row = {"text": df_source.loc[idx, "text"]}
            for t in tasks:
                col = TASK_TO_COL[t]
                row[f"{t}_true"] = int(df_source.loc[idx, col])
                row[f"{t}_prob"] = float(torch.sigmoid(outputs[t][i]).squeeze().item())
            rows.append(row)
            idx += 1

    return pd.DataFrame(rows)


# MAIN
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", choices=["2", "4"], required=True, help="Use 2-head or 4-head model.")
    p.add_argument("--ckpt_out", required=True, help="Output checkpoint .pt path")
    p.add_argument("--preds_out", required=True, help="Output test predictions .csv path")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    
    # Select Task Config
    tasks = TASKS_2 if args.tasks == "2" else TASKS_4
    print("Tasks:", tasks)

    # Load Data
    splits = load_multi_task_data(balance=BALANCE)
    train_df = splits["train"].reset_index(drop=True)
    val_df   = splits["val"].reset_index(drop=True)
    test_df  = splits["test"].reset_index(drop=True)

    print(f"Split sizes: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    # Create Datasets and DataLoaders
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = MultiTaskDataset(train_df, tokenizer, max_length=MAX_LEN, tasks=tasks)
    val_ds   = MultiTaskDataset(val_df,   tokenizer, max_length=MAX_LEN, tasks=tasks)
    test_ds  = MultiTaskDataset(test_df,  tokenizer, max_length=MAX_LEN, tasks=tasks)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Setup Loss Functions 
    pos_w = compute_pos_weights(train_df, tasks)
    print("pos_weight:", {k: round(v, 3) for k, v in pos_w.items()})

    # Create separate loss function for each task with task-specific pos_weight
    loss_fns = {t: nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w[t], device=device)) for t in tasks}

    # Initialize Model, Optimizer, and Scheduler
    model = MultiTaskRoBERTa(model_name=MODEL_NAME, tasks=tasks).to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # Linear warmup + decay learning rate schedule
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Select metric
    select_task = "Q2_harmful" if "Q2_harmful" in tasks else "Q_overall"
    best_score = -1.0

    # Training Loop
    for epoch in range(EPOCHS):
        # Train 1 epoch
        tr_loss = train_epoch(model, train_loader, tasks, device, loss_fns, optimizer, scheduler, epoch)

        # Evaluate on validation set
        va_loss, va_metrics = evaluate(model, val_loader, tasks, device, loss_fns)

        # Select validation metric for checkpoint saving
        score = va_metrics[select_task]["pr_auc"] if select_task == "Q2_harmful" else va_metrics[select_task]["f1_pos"]

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  {select_task}_score={score:.4f}")
        for t in tasks:
            m = va_metrics[t]
            print(f"  {t}: f1_pos={m['f1_pos']:.3f} pr_auc={m['pr_auc']:.3f} pred_pos_rate={m['pred_pos_rate']:.3f}")

        # Save checkpoint if validation score improved
        if score > best_score:
            best_score = score
            torch.save(
                {"model_state_dict": model.state_dict(), "tasks": tasks, "model_name": MODEL_NAME, "max_len": MAX_LEN},
                args.ckpt_out
            )
            print(f"  Saved best checkpoint to {args.ckpt_out}")

    # TEST + PREDICTIONS
    # Load best checkpoint
    ckpt = torch.load(args.ckpt_out, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate on test set
    te_loss, te_metrics = evaluate(model, test_loader, tasks, device, loss_fns)
    print(f"\nTest loss: {te_loss:.4f}")
    for t in tasks:
        m = te_metrics[t]
        print(f"{t}: acc={m['accuracy']:.3f} f1_pos={m['f1_pos']:.3f} pr_auc={m['pr_auc']:.3f}")

    # Generate and Save Test Predictions
    preds = predict(model, test_loader, test_df, tasks, device)
    preds.to_csv(args.preds_out, index=False)
    print("Saved predictions:", args.preds_out)

if __name__ == "__main__":
    main()

'''
python scripts/models/train_multitask.py \
  --tasks 4 \
  --ckpt_out models/checkpoints/best_multitask_4.pt \
  --preds_out results/predictions/multi_task_transformer/test_preds_4.csv

python scripts/models/train_multitask.py \
  --tasks 2 \
  --ckpt_out models/checkpoints/best_multitask_2.pt \
  --preds_out results/predictions/multi_task_transformer/test_preds_2.csv 
'''
