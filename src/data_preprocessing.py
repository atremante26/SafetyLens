import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

DATA_PATH = 'data/processed/dices_350_binary.csv'

TARGET_COLS = [
    'Q_overall_binary',
    'Q2_harmful_binary',
    'Q3_bias_binary',
    'Q6_policy_binary'
]

def balance_dataset(df, label_col, target_0=5000, target_1=5000, random_state=42):
    """
    Undersample a binary dataset.
    Class 0 and class 1 are undersampled to (at most) target_0/target_1.
    """
    c0 = df[df[label_col] == 0] # Safe examples
    c1 = df[df[label_col] == 1] # Unsafe examples

    # Undersample each class
    c0_bal = c0.sample(n=min(target_0, len(c0)), random_state=random_state)
    c1_bal = c1.sample(n=min(target_1, len(c1)), random_state=random_state)

    # Combine and shuffle
    out = pd.concat([c0_bal, c1_bal]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return out

def split_by_conversation(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    stratify_by: str = 'Q_overall_binary',
    balance: bool = True,
    balance_params: dict = None,
    balance_on: str = None,
):
    '''
    Split dataset by conversation (item_id) to prevent data leakage.

    Process:
    1. Group ratings by conversation (item_id)
    2. Assign each conversation to train/val/test
    3. Optionally balance each split by undersampling
    '''

    # Set default balance parameters
    if balance_params is None:
        balance_params = {"target_0": 10000, "target_1": 10000}

    # Handle legacy parameter names (target_safe/target_notsafe)
    if ("target_safe" in balance_params) or ("target_notsafe" in balance_params):
        balance_params = {
            "target_0": balance_params.get("target_safe", balance_params.get("target_0", 10000)),
            "target_1": balance_params.get("target_notsafe", balance_params.get("target_1", 10000)),
        }

    # Default: balance on the same label used for stratification
    if balance_on is None and balance:
        balance_on = stratify_by

    # Get one label per conversation
    conv_labels = (
        df.groupby('item_id')[stratify_by]
          .agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
          .reset_index()
    )

    unique_convs = conv_labels['item_id'].values
    labels = conv_labels[stratify_by].values

    # Split conversations into train + val and test
    train_val_convs, test_convs, train_val_labels, _ = train_test_split(
        unique_convs,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Split train + val into train and val
    val_proportion = val_size / (train_size + val_size)
    train_convs, val_convs = train_test_split(
        train_val_convs,
        test_size=val_proportion,
        random_state=random_state,
        stratify=train_val_labels
    )

    # Map conversations back to individual ratings
    splits = {
        "train": df[df["item_id"].isin(train_convs)].copy(),
        "val":   df[df["item_id"].isin(val_convs)].copy(),
        "test":  df[df["item_id"].isin(test_convs)].copy(),
    }

    # Optionally balance each split by undersapling
    if balance:
        t0 = balance_params.get("target_0", 10000)
        t1 = balance_params.get("target_1", 10000)

        # Use smaller targets for val/test (20% of train targets)
        val_t0, val_t1 = int(t0 * 0.2), int(t1 * 0.2)
        test_t0, test_t1 = int(t0 * 0.2), int(t1 * 0.2)

        splits["train"] = balance_dataset(splits["train"], balance_on, t0, t1, random_state)
        splits["val"]   = balance_dataset(splits["val"],   balance_on, val_t0, val_t1, random_state)
        splits["test"]  = balance_dataset(splits["test"],  balance_on, test_t0, test_t1, random_state)

    return splits

# DATA LOADERS
def load_data_sklearn(
    target: str = 'Q_overall_binary',
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    balance: bool = True,
    balance_params: dict = None,
    max_features: int = 20000,
    ngram_range: tuple = (1, 2)
):
    """
    Load data for sklearn models (Logistic Regression) using TF-IDF
    """
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Split by conversation
    splits = split_by_conversation(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify_by=target,
        balance=balance,
        balance_params=balance_params
    )

    # Extract text and labels
    X_train_text = splits['train']['text'].values
    X_val_text   = splits['val']['text'].values
    X_test_text  = splits['test']['text'].values

    y_train = splits['train'][target].values
    y_val   = splits['val'][target].values
    y_test  = splits['test'][target].values

    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        stop_words='english'
    )

    X_train = vectorizer.fit_transform(X_train_text)
    X_val = vectorizer.transform(X_val_text)
    X_test = vectorizer.transform(X_test_text)

    # Save vectorizer (REQUIRED for SHAP)
    vec_filepath = 'models/checkpoints/tfidf_vectorizer.pkl'

    with open(vec_filepath, "wb") as f:
        pickle.dump(vectorizer, f)

    return X_train, X_val, X_test, y_train, y_val, y_test

def load_data_transformers(
    target: str = 'Q_overall_binary',
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    balance: bool = True,
    balance_params: dict = None
):
    """
    Load data for single-task transformer models (binary classification)
    """
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Split by conversation 
    splits = split_by_conversation(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify_by=target,
        balance=balance,
        balance_params=balance_params
    )
    
    # Keep only text and target label, rename for HuggingFace compatibility
    for split_name in ['train', 'val', 'test']:
        splits[split_name] = splits[split_name][['text', target]].copy()
        splits[split_name].rename(columns={target: 'label'}, inplace=True)

    return splits


def load_multi_task_data(
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    balance: bool = False,
    balance_params: dict = None
):
    """
    Load data for multi-task transformer (predicts all 2 or 4 binary targets simultaneously)
    """
    # Load dataset
    df = pd.read_csv(DATA_PATH)
    
    # Split by conversation 
    splits = split_by_conversation(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify_by='Q_overall_binary',  # Use overall safety for stratification
        balance=balance,
        balance_params=balance_params,
        balance_on=None
    )
    
    # Keep text + all 4 binary targets
    for split_name in ['train', 'val', 'test']:
        splits[split_name] = splits[split_name][['text'] + TARGET_COLS].copy()
    
    return splits


if __name__ == '__main__':
    # Test sklearn loader
    print("Testing sklearn loader...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_sklearn()
    print(f"sklearn loader works: {len(X_train):,} train samples\n")
    
    # Test multi-task loader
    print("Testing multi-task loader...")
    splits = load_multi_task_data()
    print(f"Multi-task loader works:")
    print(f"  Train: {len(splits['train']):,} ratings")
    print(f"  Val: {len(splits['val']):,} ratings")
    print(f"  Test: {len(splits['test']):,} ratings")