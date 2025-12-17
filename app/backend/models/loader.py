"""
Model loader for all SafetyLens models.
Loads models once on startup and provides prediction interfaces.
"""

import pickle
import torch
import importlib.util
from pathlib import Path
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from typing import Optional, Dict, Any

# Get absolute path to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "checkpoints"

print(f"📁 PROJECT_ROOT: {PROJECT_ROOT}")

# Direct import to avoid package name conflicts
multi_task_path = PROJECT_ROOT / "models" / "multi_task_transformer.py"

if not multi_task_path.exists():
    raise FileNotFoundError(f"Cannot find: {multi_task_path}")

# Load module directly
spec = importlib.util.spec_from_file_location("multi_task_transformer", multi_task_path)
multi_task_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_task_module)

# Extract what we need
MultiTaskRoBERTa = multi_task_module.MultiTaskRoBERTa
load_model = multi_task_module.load_model

print("✓ Imported MultiTaskRoBERTa and load_model")

class ModelLoader:
    """Centralized model loader for all SafetyLens models"""
    
    def __init__(self):
        # Logistic Regression
        self.logreg_model = None
        self.vectorizer = None
        
        # Transformers
        self.singletask_model = None
        self.singletask_tokenizer = None
        
        self.multitask_2_model = None
        self.multitask_2_tokenizer = None
        
        self.multitask_4_model = None
        self.multitask_4_tokenizer = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_all_models(self):
        """Load all models on startup"""
        self.load_logreg()
        self.load_singletask()
        self.load_multitask_2()
        self.load_multitask_4()
    
    def load_logreg(self):
        """Load logistic regression + TF-IDF vectorizer"""
        try:
            logreg_path = MODELS_DIR / "logistic_regression_model.pkl"
            vec_path = MODELS_DIR / "tfidf_vectorizer.pkl"
            
            with open(logreg_path, "rb") as f:
                self.logreg_model = pickle.load(f)
            
            with open(vec_path, "rb") as f:
                self.vectorizer = pickle.load(f)
            
            print("✓ Loaded Logistic Regression model")
        except Exception as e:
            print(f"✗ Failed to load Logistic Regression: {e}")
    
    def load_singletask(self):
        """Load single-task RoBERTa model"""
        try:
            model_path = MODELS_DIR / "best_singletask.pt"
            
            if not model_path.exists():
                print(f"⚠ Single-task checkpoint not found: {model_path}")
                return
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize model
            self.singletask_model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=2
            )
            
            # The checkpoint IS the state dict directly (not wrapped in a dict)
            # Just load it directly
            self.singletask_model.load_state_dict(checkpoint, strict=False)
            
            self.singletask_model.to(self.device)
            self.singletask_model.eval()
            
            # Load tokenizer
            self.singletask_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            
            print("✓ Loaded Single-Task model")
            
        except Exception as e:
            print(f"✗ Failed to load Single-Task model: {e}")
            import traceback
            traceback.print_exc()
    
    def load_multitask_2(self):
        """Load 2-head multi-task model"""
        try:
            model_path = MODELS_DIR / "best_multitask_2.pt"
            
            if not model_path.exists():
                print(f"⚠ Multi-Task-2 checkpoint not found: {model_path}")
                return
            
            self.multitask_2_model, self.multitask_2_tokenizer = load_model(
                str(model_path), 
                self.device
            )
            
            print(f"✓ Loaded Multi-Task-2 model (tasks: {self.multitask_2_model.tasks})")
            
        except Exception as e:
            print(f"✗ Failed to load Multi-Task-2 model: {e}")
            import traceback
            traceback.print_exc()
    
    def load_multitask_4(self):
        """Load 4-head multi-task model"""
        try:
            model_path = MODELS_DIR / "best_multitask_4.pt"
            
            if not model_path.exists():
                print(f"⚠ Multi-Task-4 checkpoint not found: {model_path}")
                return
            
            self.multitask_4_model, self.multitask_4_tokenizer = load_model(
                str(model_path), 
                self.device
            )
            
            print(f"✓ Loaded Multi-Task-4 model (tasks: {self.multitask_4_model.tasks})")
            
        except Exception as e:
            print(f"✗ Failed to load Multi-Task-4 model: {e}")
            import traceback
            traceback.print_exc()
    
    def predict_logreg(self, text: str) -> Dict[str, Any]:
        """Predict with logistic regression"""
        if self.logreg_model is None or self.vectorizer is None:
            raise ValueError("Logistic regression model not loaded")
        
        X = self.vectorizer.transform([text])
        pred = self.logreg_model.predict(X)[0]
        proba = self.logreg_model.predict_proba(X)[0]
        
        return {
            "prediction": int(pred),
            "probability": float(proba[1]),
            "label": "Unsafe" if pred == 1 else "Safe"
        }
    
    def predict_singletask(self, text: str) -> Dict[str, Any]:
        """Predict with single-task transformer"""
        if self.singletask_model is None or self.singletask_tokenizer is None:
            raise ValueError("Single-task model not loaded")
        
        inputs = self.singletask_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.singletask_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            prob_unsafe = probs[0][1].item()
        
        return {
            "prediction": int(pred),
            "probability": float(prob_unsafe),
            "label": "Unsafe" if pred == 1 else "Safe"
        }
    
    def predict_multitask(self, text: str, task: str = "Q_overall", num_heads: int = 2) -> Dict[str, Any]:
        """Predict with multi-task transformer"""
        
        if num_heads == 2:
            model = self.multitask_2_model
            tokenizer = self.multitask_2_tokenizer
        elif num_heads == 4:
            model = self.multitask_4_model
            tokenizer = self.multitask_4_tokenizer
        else:
            raise ValueError(f"Invalid num_heads: {num_heads}. Must be 2 or 4.")
        
        if model is None or tokenizer is None:
            raise ValueError(f"Multi-task-{num_heads} model not loaded")
        
        if task not in model.tasks:
            raise ValueError(f"Task '{task}' not available. Available: {model.tasks}")
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding="max_length"
        ).to(self.device)
        
        with torch.no_grad():
            logits_dict = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            logit = logits_dict[task]
            prob_unsafe = torch.sigmoid(logit).squeeze().item()
            pred = 1 if prob_unsafe > 0.5 else 0
        
        return {
            "prediction": int(pred),
            "probability": float(prob_unsafe),
            "label": "Unsafe" if pred == 1 else "Safe",
            "task": task
        }