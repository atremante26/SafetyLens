import sys
from pathlib import Path
import argparse
import random
from typing import Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, average_precision_score
)

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from models.single_task_transformer import SingleTaskDataset, load_model, MODEL_NAME
from src.data_preprocessing import load_data_sklearn

# HYPERPARAMETERS
SEED = 42
MAX_LEN = 256
BATCH_SIZE = 16
EPOCHS = 3
LR = 2e-5
WARMUP_FRAC = 0.1
NUM_WORKERS = 0


# UTILS
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute comprehensive binary classification metrics.
    """
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


def compute_pos_weight(train_df: pd.DataFrame, label_col: str = "Q_overall_binary"):
    """
    Calculate pos_weight for BCEWithLogitsLoss to handle class imbalance.
    
    pos_weight = (# negative examples) / (# positive examples)
    """
    y = train_df[label_col].astype(int).values
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return float(neg / max(pos, 1))


# TRAIN / EVAL / PRED
def train_epoch(model, loader, device, criterion, optimizer, scheduler, epoch_idx):
    """
    Train model for one epoch.
    """
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc=f"Train epoch {epoch_idx+1}", leave=False):
        # Move batch to device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    """
    Evaluate model on validation or test set.
    """
    model.eval()
    running_loss = 0.0
    all_true = []
    all_prob = []

    for batch in tqdm(loader, desc="Eval", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)[:, 1]  # P(unsafe)

        # Collect predictions
        all_prob.extend(probs.cpu().numpy().tolist())
        all_true.extend(labels.cpu().numpy().tolist())

        running_loss += loss.item()

    # Compute metrics
    metrics = compute_binary_metrics(all_true, all_prob)
    avg_loss = running_loss / max(len(loader), 1)

    return avg_loss, metrics


@torch.no_grad()
def predict(model, loader, df_source, device):
    """
    Generate predictions on test set.
    
    Creates a dataframe with:
    - Original text
    - Ground truth label (Q_overall_true)
    - Predicted probability (Q_overall_prob)
    """
    model.eval()
    rows = []
    df_source = df_source.reset_index(drop=True)
    idx = 0

    for batch in tqdm(loader, desc="Predict", leave=False):
        bs = batch["input_ids"].shape[0]
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)

        # Build prediction row for each example in batch
        for i in range(bs):
            row = {
                "text": df_source.loc[idx, "text"],
                "Q_overall_true": int(df_source.loc[idx, "Q_overall_binary"]),
                "Q_overall_prob": float(probs[i, 1].item()),  # P(unsafe)
            }
            rows.append(row)
            idx += 1

    return pd.DataFrame(rows)


# MAIN
def parse_args():
    p = argparse.ArgumentParser(description="Train single-task RoBERTa for safety classification")
    p.add_argument("--ckpt_out", required=True, help="Output checkpoint .pt path")
    p.add_argument("--preds_out", required=True, help="Output test predictions .csv path")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Load Data
    print("Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_sklearn()
    
    # Create DataFrames for dataset creation
    train_df = pd.DataFrame({"text": X_train, "Q_overall_binary": y_train})
    val_df = pd.DataFrame({"text": X_val, "Q_overall_binary": y_val})
    test_df = pd.DataFrame({"text": X_test, "Q_overall_binary": y_test})

    print(f"Split sizes: train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}")

    # Create Datasets and DataLoaders
    print("Creating datasets...")
    model, tokenizer = load_model(model_name=MODEL_NAME, device=device)
    
    train_ds = SingleTaskDataset(train_df, tokenizer, label_col="Q_overall_binary", max_length=MAX_LEN)
    val_ds = SingleTaskDataset(val_df, tokenizer, label_col="Q_overall_binary", max_length=MAX_LEN)
    test_ds = SingleTaskDataset(test_df, tokenizer, label_col="Q_overall_binary", max_length=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # Setup Loss Function with class imbalance handling
    pos_weight = compute_pos_weight(train_df)
    print(f"pos_weight: {pos_weight:.3f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))

    # Initialize Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=LR)

    # Linear warmup + decay learning rate schedule
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_FRAC * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training Loop
    best_score = -1.0
    print("\nStarting training...")

    for epoch in range(EPOCHS):
        # Train one epoch
        tr_loss = train_epoch(model, train_loader, device, criterion, optimizer, scheduler, epoch)

        # Evaluate on validation set
        va_loss, va_metrics = evaluate(model, val_loader, device, criterion)

        # Use F1 score as selection metric
        score = va_metrics["f1_pos"]

        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print(f"  train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}  val_f1={score:.4f}")
        print(f"  accuracy={va_metrics['accuracy']:.3f} precision={va_metrics['precision_pos']:.3f} "
              f"recall={va_metrics['recall_pos']:.3f} pr_auc={va_metrics['pr_auc']:.3f}")

        # Save checkpoint if validation score improved
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": MODEL_NAME,
                    "max_len": MAX_LEN,
                },
                args.ckpt_out
            )
            print(f"Saved best checkpoint to {args.ckpt_out}")

    # TEST + PREDICTIONS
    print("\nEvaluating on test set...")
    
    # Load best checkpoint
    ckpt = torch.load(args.ckpt_out, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Evaluate on test set
    te_loss, te_metrics = evaluate(model, test_loader, device, criterion)
    print(f"\nTest Results:")
    print(f"  loss={te_loss:.4f}")
    print(f"  accuracy={te_metrics['accuracy']:.3f} precision={te_metrics['precision_pos']:.3f} "
          f"recall={te_metrics['recall_pos']:.3f}")
    print(f"  f1_pos={te_metrics['f1_pos']:.3f} pr_auc={te_metrics['pr_auc']:.3f}")

    # Generate and Save Test Predictions
    print("\nGenerating predictions...")
    preds = predict(model, test_loader, test_df, device)
    preds.to_csv(args.preds_out, index=False)
    print(f"Saved predictions: {args.preds_out}")


if __name__ == "__main__":
    main()