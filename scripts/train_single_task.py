import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_scheduler
import pandas as pd
from sklearn.model_selection import train_test_split
from models.single_task_transformer import HateSpeechDataset, load_model, load_finetuned, MODEL_NAME, label2id

#https://huggingface.co/learn/llm-course/chapter3/4

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_correct = 0
    total = 0
    total_loss = 0.0

    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        logits = outputs.logits

        preds = torch.argmax(logits, dim=-1)
        total_correct += (preds == batch["labels"]).sum().item()
        total += batch["labels"].size(0)
        total_loss += loss.item() * batch["labels"].size(0)

    avg_loss = total_loss / max(total, 1)
    acc = total_correct / max(total, 1)
    model.train()
    return avg_loss, acc


def train(train_df,val_df,batch_size,lr,epochs,max_length,output_dir="results"):
    random.seed(67)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model, tokenizer = load_model(model_name=MODEL_NAME, device=device)

    print("Loading datasets...")
    train_ds = HateSpeechDataset(train_df, tokenizer,label_col="Q_overall_binary", max_length=max_length)
    val_ds = HateSpeechDataset(val_df, tokenizer,label_col="Q_overall_binary", max_length=max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    num_training_steps = epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    print("begining training loop...")
    global_step = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            global_step += 1

        train_loss = running_loss / max(len(train_loader), 1)
        val_loss, val_acc = evaluate(model, val_loader, device)

        print(
            f"Epoch {epoch+1}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.4f}"
        )

    model_path = os.path.join(output_dir, "roberta_single_task.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

    return model_path


def run_predictions(
    df,
    model_path,
    output_csv,
    text_col="text",
    label_col="Q_overall_binary",
    batch_size=32,
    max_length=128,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("loading model")
    # Load model + tokenizer
    model, tokenizer = load_finetuned(
        model_path="roberta_single_task.pt",
        device=device
    )
    model.eval()
    print("loading data")
    # Dataset + DataLoader
    dataset = HateSpeechDataset(
