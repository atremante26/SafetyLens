import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

DATA_PATH = '../data/processed/dices_350_binary.csv'

TARGET_COLS = [
    'Q_overall_binary',
    'Q2_harmful_binary',
    'Q3_bias_binary',
    'Q6_policy_binary'
]


def balance_dataset(df, target_safe=5000, target_notsafe=5000):
    """
    Balance dataset for BINARY classification: Safe (0) vs Not Safe (1)
    
    Args:
        df: DataFrame with Q_overall_binary column
        target_safe: Number of Safe examples to keep
        target_notsafe: Number of Not Safe examples to keep
    
    Returns:
        Balanced DataFrame
    """
    # Separate by binary class
    safe = df[df['Q_overall_binary'] == 0]
    not_safe = df[df['Q_overall_binary'] == 1]
    
    # Undersample both classes to target counts
    safe_balanced = safe.sample(n=min(target_safe, len(safe)), random_state=42)
    notsafe_balanced = not_safe.sample(n=min(target_notsafe, len(not_safe)), random_state=42)
    
    # Combine and shuffle
    balanced_df = pd.concat([safe_balanced, notsafe_balanced])
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    
    return balanced_df


def split_by_conversation(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    stratify_by: str = 'Q_overall_binary',
    balance: bool = True,
    balance_params: dict = None
):
    """
    Split data by conversation (item_id) to prevent data leakage,
    but return ALL individual ratings for training
    
    Args:
        df: DataFrame with 'item_id', 'text', and target columns
        train_size: Proportion for training (default 0.7)
        val_size: Proportion for validation (default 0.2)
        test_size: Proportion for test (default 0.1)
        random_state: Random seed for reproducibility
        stratify_by: Target variable to use for stratification (default: Q_overall_binary)
        balance: If True, apply undersampling to balance classes (default True)
        balance_params: Dict with 'target_safe' and 'target_notsafe' for balancing
        
    Returns:
        Dictionary with keys 'train', 'val', 'test'
        Each contains individual ratings from non-overlapping conversations
    """
    
    # Get unique conversations with their majority labels
    conv_labels = df.groupby('item_id')[stratify_by].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    ).reset_index()
    
    unique_convs = conv_labels['item_id'].values
    labels = conv_labels[stratify_by].values
    
    # Split conversations into train+val / test
    train_val_convs, test_convs, train_val_labels, test_labels = train_test_split(
        unique_convs, 
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels
    )
    
    # Split train+val into train / val
    val_proportion = val_size / (train_size + val_size)
    train_convs, val_convs = train_test_split(
        train_val_convs,
        test_size=val_proportion,
        random_state=random_state,
        stratify=train_val_labels
    )
    
    # Get all individual ratings for each conversation set
    splits = {
        'train': df[df['item_id'].isin(train_convs)].copy(),
        'val': df[df['item_id'].isin(val_convs)].copy(),
        'test': df[df['item_id'].isin(test_convs)].copy()
    }
    
    # Apply balancing if requested
    if balance:
        if balance_params is None:
            balance_params = {'target_safe': 10000, 'target_notsafe': 10000}
        
        # Train
        splits['train'] = balance_dataset(
            splits['train'], 
            target_safe=balance_params['target_safe'],
            target_notsafe=balance_params['target_notsafe']
        )
        
        # Validation
        val_safe = int(balance_params['target_safe'] * 0.2)
        val_notsafe = int(balance_params['target_notsafe'] * 0.2)
        splits['val'] = balance_dataset(
            splits['val'],
            target_safe=val_safe,
            target_notsafe=val_notsafe
        )
        
        # Test
        test_safe = int(balance_params['target_safe'] * 0.2)
        test_notsafe = int(balance_params['target_notsafe'] * 0.2)
        splits['test'] = balance_dataset(
            splits['test'],
            target_safe=test_safe,
            target_notsafe=test_notsafe
        )
    
    return splits


def load_data_sklearn(
    target: str = 'Q_overall_binary',
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 2,
    workers: int = 4,
    balance: bool = True,
    balance_params: dict = None
):
    """
    Load data for sklearn models (Logistic Regression) with binary classification
    Trains on individual ratings from non-overlapping conversations
    """
    # Load individual ratings
    df = pd.read_csv(DATA_PATH)
    
    # Split by conversation (returns individual ratings)
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
    X_val_text = splits['val']['text'].values
    X_test_text = splits['test']['text'].values
    
    y_train = splits['train'][target].values
    y_val = splits['val'][target].values
    y_test = splits['test'][target].values

    # Tokenize
    train_tokens = [simple_preprocess(text) for text in X_train_text]
    val_tokens = [simple_preprocess(text) for text in X_val_text]
    test_tokens = [simple_preprocess(text) for text in X_test_text]

    # Train Word2Vec on training data
    w2v_model = Word2Vec(
        sentences=train_tokens,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        seed=random_state,
        epochs=10
    )

    def text_to_vector(tokens, model):
        """Average word vectors for all words in the text"""
        vectors = []
        for word in tokens:
            if word in model.wv:
                vectors.append(model.wv[word])
        
        if len(vectors) > 0:
            return np.mean(vectors, axis=0)
        else:
            return np.zeros(model.vector_size)

    # Convert all texts to vectors
    X_train = np.array([text_to_vector(tokens, w2v_model) for tokens in train_tokens])
    X_val = np.array([text_to_vector(tokens, w2v_model) for tokens in val_tokens])
    X_test = np.array([text_to_vector(tokens, w2v_model) for tokens in test_tokens])
    
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
    Trains on individual ratings from non-overlapping conversations
    """
    # Load individual ratings
    df = pd.read_csv(DATA_PATH)
    
    # Split by conversation (returns individual ratings)
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
    
    # Rename target column to 'label' for HuggingFace compatibility
    for split_name in ['train', 'val', 'test']:
        splits[split_name] = splits[split_name][['text', target]].copy()
        splits[split_name].rename(columns={target: 'label'}, inplace=True)

    return splits


def load_multi_task_data(
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    balance: bool = True,
    balance_params: dict = None
):
    """
    Load data for multi-task transformer (predicts all 4 binary targets simultaneously)
    Trains on individual ratings from non-overlapping conversations
    """
    # Load individual ratings
    df = pd.read_csv(DATA_PATH)
    
    # Split by conversation (returns individual ratings)
    splits = split_by_conversation(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify_by='Q_overall_binary',  # Use overall safety for stratification
        balance=balance,
        balance_params=balance_params
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