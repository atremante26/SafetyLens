import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = '../data/processed/individual_ratings_3class.csv'

TARGET_COLS = [
    'Q_overall_3class',
    'Q2_harmful_content_overall_3class',
    'Q3_bias_overall_3class',
    'Q6_policy_guidelines_overall_3class'
]


def split_by_conversation(
    df: pd.DataFrame,
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42,
    stratify_by: str = 'Q_overall_3class'
):
    """
    Split data by conversation (item_id) to prevent data leakage.
    
    Args:
        df: DataFrame with 'item_id', 'text', and target columns
        train_size: Proportion for training (default 0.7)
        val_size: Proportion for validation (default 0.2)
        test_size: Proportion for test (default 0.1)
        random_state: Random seed for reproducibility
        stratify_by: Target variable to use for stratification
        
    Returns:
        Dictionary with keys 'train', 'val', 'test'
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
    
    # Get all ratings for each conversation set
    splits = {
        'train': df[df['item_id'].isin(train_convs)].copy(),
        'val': df[df['item_id'].isin(val_convs)].copy(),
        'test': df[df['item_id'].isin(test_convs)].copy()
    }

    return splits


def load_data_sklearn(
    target: str = 'Q_overall_3class',
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42
):
    """
    Load data for sklearn models (Logistic Regression).
    
    Args:
        target: Which target variable to predict
        train_size, val_size, test_size: Split proportions
        random_state: Random seed
        
    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
        (X are text strings, y are integer labels)
    """
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Split by conversation
    splits = split_by_conversation(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify_by=target
    )
    
    # Extract text and labels
    X_train = splits['train']['text'].values
    X_val = splits['val']['text'].values
    X_test = splits['test']['text'].values
    
    y_train = splits['train'][target].values
    y_val = splits['val'][target].values
    y_test = splits['test'][target].values
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def load_data_transformers(
    target: str = 'Q_overall_3class',
    train_size: float = 0.7,
    val_size: float = 0.2,
    test_size: float = 0.1,
    random_state: int = 42
):
    """
    Load data for single-task transformer models.
    
    Args:
        target: Which target variable to predict
        train_size, val_size, test_size: Split proportions
        random_state: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
        Each DataFrame has 'text' and 'label' columns
    """
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Split by conversation
    splits = split_by_conversation(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify_by=target
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
    random_state: int = 42
):
    """
    Load data for multi-task transformer (predicts all 4 targets simultaneously).
    
    Args:
        train_size, val_size, test_size: Split proportions
        random_state: Random seed
        
    Returns:
        Dictionary with 'train', 'val', 'test' DataFrames
        Each DataFrame has 'text' and all 4 target columns
    """
    # Load data
    df = pd.read_csv(DATA_PATH)
    
    # Split by conversation
    splits = split_by_conversation(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
        stratify_by='Q_overall_3class'  # Use overall safety as stratification
    )
    
    # Keep text + all 4 targets
    for split_name in ['train', 'val', 'test']:
        splits[split_name] = splits[split_name][['text'] + TARGET_COLS].copy()
    
    return splits


if __name__ == '__main__':
    # Testing dataloaders
    print("Testing data loaders...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_data_sklearn()
    print(f"sklearn loader works: {len(X_train):,} train samples")