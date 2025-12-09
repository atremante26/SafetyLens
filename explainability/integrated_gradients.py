import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from captum.attr import IntegratedGradients


def prepare_input(text, tokenizer, device):
    """
    Tokenize text and prepare for model input.
    
    Args:
        text: Input text string
        tokenizer: RoBERTa tokenizer
        device: torch device (cuda/cpu)
    
    Returns:
        Dictionary with input_ids, attention_mask, and tokens
    """
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return {
        'input_ids': encoding['input_ids'].to(device),
        'attention_mask': encoding['attention_mask'].to(device),
        'tokens': tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    }


def get_prediction(model, input_ids, attention_mask, task='Q_overall'):
    """
    Get model prediction for a specific task.
    
    Args:
        model: Multi-task model
        input_ids: Tokenized input
        attention_mask: Attention mask
        task: Task name ('Q_overall', 'Q2_harmful', 'Q3_bias', 'Q6_policy')
    
    Returns:
        Dictionary with predicted class, probabilities, and class names
    """
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs[task]
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1)
    
    return {
        'predicted_class': pred_class.item(),
        'probabilities': probs[0].cpu().numpy(),
        'class_names': ['Safe', 'Ambiguous', 'Unsafe']
    }


def construct_baseline(input_ids, attention_mask, baseline_token_id=1):
    """
    Construct baseline (reference) input for Integrated Gradients.
    Baseline is all padding tokens.
    
    Args:
        input_ids: Original input token ids
        attention_mask: Original attention mask
        baseline_token_id: Token ID for baseline (1 = <pad> for RoBERTa)
    
    Returns:
        Tuple of (input, baseline)
    """
    ref_input_ids = torch.ones_like(input_ids) * baseline_token_id
    ref_attention_mask = torch.zeros_like(attention_mask)
    
    return (input_ids, attention_mask), (ref_input_ids, ref_attention_mask)


def compute_ig_attributions(text, model, tokenizer, task='Q_overall', n_steps=50, device='cuda'):
    """
    Compute Integrated Gradients attributions for a text input.
    
    Args:
        text: Input text string
        model: Multi-task model
        tokenizer: RoBERTa tokenizer
        task: Which task to analyze ('Q_overall', 'Q2_harmful', 'Q3_bias', 'Q6_policy')
        n_steps: Number of steps for IG approximation (higher = more accurate)
        device: torch device
    
    Returns:
        Dictionary with:
            - text: Original input text
            - task: Task analyzed
            - tokens: List of tokens
            - attributions: Attribution scores per token
            - predicted_class: Predicted class index
            - class_name: Predicted class name
            - probabilities: Prediction probabilities
            - convergence_delta: IG convergence metric
            - metrics: Attribution statistics
    """
    # Prepare input
    inputs = prepare_input(text, tokenizer, device)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']
    tokens = inputs['tokens']
    
    # Get prediction
    prediction = get_prediction(model, input_ids, attention_mask, task)
    predicted_class = prediction['predicted_class']
    
    # Construct baseline
    (input_ids, attention_mask), (ref_input_ids, ref_attention_mask) = \
        construct_baseline(input_ids, attention_mask)
    
    # Define forward function for this task
    def forward_func(ids, mask):
        outputs = model(ids, mask)
        return outputs[task]
    
    # Initialize Integrated Gradients
    ig = IntegratedGradients(forward_func)
    
    # Compute attributions
    attributions, delta = ig.attribute(
        inputs=(input_ids, attention_mask),
        baselines=(ref_input_ids, ref_attention_mask),
        target=predicted_class,
        n_steps=n_steps,
        return_convergence_delta=True
    )
    
    # Extract token attributions (sum over embedding dimensions)
    token_attributions = attributions[0].sum(dim=2).squeeze(0)
    token_attributions = token_attributions.cpu().detach().numpy()
    
    # Normalize attributions
    token_attributions = token_attributions / np.linalg.norm(token_attributions)
    
    return {
        'text': text,
        'task': task,
        'tokens': tokens,
        'attributions': token_attributions,
        'predicted_class': predicted_class,
        'class_name': prediction['class_names'][predicted_class],
        'probabilities': prediction['probabilities'],
        'convergence_delta': delta.item(),
        'metrics': compute_attribution_metrics(token_attributions)
    }


def compute_attribution_metrics(attributions):
    """
    Compute statistical metrics for attribution analysis.
    
    Args:
        attributions: Array of attribution scores
    
    Returns:
        Dictionary with mean, std, max, and sparsity metrics
    """
    non_zero_attrs = attributions[attributions != 0]
    
    if len(non_zero_attrs) == 0:
        return {'mean': 0, 'std': 0, 'max': 0, 'sparsity': 1.0}
    
    return {
        'mean': float(np.mean(np.abs(non_zero_attrs))),
        'std': float(np.std(non_zero_attrs)),
        'max': float(np.max(np.abs(non_zero_attrs))),
        'sparsity': float(1.0 - (len(non_zero_attrs) / len(attributions)))
    }


def get_top_tokens(attributions, tokens, top_k=10):
    """
    Get top-k most influential tokens.
    
    Args:
        attributions: Attribution scores
        tokens: List of tokens
        top_k: Number of top tokens to return
    
    Returns:
        List of (token, attribution) tuples sorted by absolute attribution
    """
    special_tokens = ['<s>', '</s>', '<pad>']
    filtered = [
        (token, attr) for token, attr in zip(tokens, attributions)
        if token not in special_tokens and attr != 0
    ]
    
    sorted_attrs = sorted(filtered, key=lambda x: abs(x[1]), reverse=True)
    return sorted_attrs[:top_k]


def visualize_attributions(result, save_path=None, figsize=(14, 6)):
    """
    Visualize token attributions as a bar chart.
    
    Args:
        result: Result dictionary from compute_ig_attributions()
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    tokens = result['tokens']
    attributions = result['attributions']
    
    # Filter out padding and special tokens
    special_tokens = ['<s>', '</s>', '<pad>']
    filtered_data = [
        (token, attr) for token, attr in zip(tokens, attributions)
        if token not in special_tokens and attr != 0
    ]
    
    if len(filtered_data) == 0:
        print("No attributions to visualize")
        return
    
    tokens_filtered, attrs_filtered = zip(*filtered_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color: red for negative (decreases prediction), green for positive (increases)
    colors = ['red' if attr < 0 else 'green' for attr in attrs_filtered]
    
    # Bar plot
    ax.bar(range(len(tokens_filtered)), attrs_filtered, color=colors, alpha=0.6)
    
    # Customize
    ax.set_xticks(range(len(tokens_filtered)))
    ax.set_xticklabels(tokens_filtered, rotation=45, ha='right', fontsize=10)
    ax.set_ylabel('Attribution Score', fontsize=12)
    ax.set_title(
        f"Token Attributions - {result['task']} (Predicted: {result['class_name']})",
        fontsize=14,
        fontweight='bold'
    )
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def visualize_top_tokens(result, top_k=10, save_path=None, figsize=(10, 6)):
    """
    Visualize top-k most influential tokens as horizontal bar chart.
    
    Args:
        result: Result dictionary from compute_ig_attributions()
        top_k: Number of top tokens to show
        save_path: Optional path to save figure
        figsize: Figure size tuple
    """
    top_tokens = get_top_tokens(result['attributions'], result['tokens'], top_k)
    
    if len(top_tokens) == 0:
        print("No tokens to visualize")
        return
    
    tokens, attrs = zip(*top_tokens)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = ['red' if attr < 0 else 'green' for attr in attrs]
    
    ax.barh(range(len(tokens)), attrs, color=colors, alpha=0.7)
    ax.set_yticks(range(len(tokens)))
    ax.set_yticklabels(tokens, fontsize=11)
    ax.set_xlabel('Attribution Score', fontsize=12)
    ax.set_title(
        f"Top {top_k} Influential Tokens - {result['task']}",
        fontsize=14,
        fontweight='bold'
    )
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Invert y-axis so most influential is on top
    ax.invert_yaxis()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()


def analyze_hypothesis_h4(results, toxic_keywords=None):
    """
    Test H4: Toxic keywords receive high attribution.
    
    Args:
        results: Dictionary with category -> list of result dicts
        toxic_keywords: List of toxic keywords to check
    
    Returns:
        Dictionary with analysis results
    """
    if toxic_keywords is None:
        toxic_keywords = ['idiot', 'stupid', 'dumb', 'hate', 'terrible', 'awful', 
                         'worst', 'horrible', 'disgusting', 'pathetic']
    
    analysis = {}
    
    for category in results:
        analysis[category] = []
        
        for result in results[category]:
            top_tokens = get_top_tokens(result['attributions'], result['tokens'], 10)
            top_words = [token.lower().strip('Ġ') for token, _ in top_tokens]
            
            toxic_in_top = [word for word in top_words if word in toxic_keywords]
            
            analysis[category].append({
                'text': result['text'][:50] + '...',
                'top_tokens': top_words[:5],
                'toxic_keywords_found': toxic_in_top,
                'has_toxic_keyword': len(toxic_in_top) > 0
            })
    
    return analysis


def analyze_hypothesis_h5(results):
    """
    Test H5: Ambiguous examples have more diffuse attribution (higher sparsity).
    
    Args:
        results: Dictionary with category -> list of result dicts
    
    Returns:
        Dictionary with sparsity statistics per category
    """
    sparsity_by_category = {}
    
    for category in results:
        sparsities = [r['metrics']['sparsity'] for r in results[category]]
        sparsity_by_category[category] = {
            'mean': np.mean(sparsities),
            'std': np.std(sparsities),
            'values': sparsities
        }
    
    return sparsity_by_category