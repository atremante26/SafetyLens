import os
import gc
import sys
import torch
import pickle
from pathlib import Path
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import logging

logger = logging.getLogger(__name__)

# Detect if running on Render vs locally
IS_RENDER = os.getenv('RENDER') is not None

# Set project root
if IS_RENDER:
    # On Render: /opt/render/project/src/app/backend/models/loader.py
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
else:
    # Locally: SafetyLens/app/backend/models/loader.py
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.multi_task_transformer import MultiTaskRoBERTa

class ModelLoader:
    """Handles loading and management of all models with lazy loading support"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model paths
        self.models_dir = PROJECT_ROOT / "models" / "checkpoints"
        
        # Initialize all models as None (lazy loading)
        self.logreg_model = None
        self.tfidf_vectorizer = None
        self.singletask_model = None
        self.singletask_tokenizer = None
        self.multitask_2_model = None
        self.multitask_2_tokenizer = None
        self.multitask_4_model = None
        self.multitask_4_tokenizer = None
        
        # Track which model is currently loaded (for memory management)
        self.currently_loaded_transformer = None

    def unload_transformers(self, keep_model=None):
        """Unload transformer models to free memory, optionally keeping one"""
        
        if keep_model != 'singletask' and self.singletask_model is not None:
            logger.info("Unloading Single-Task model to free memory")

            # Delete references to allow garbage collection
            del self.singletask_model
            del self.singletask_tokenizer

            # Set attributes to None 
            self.singletask_model = None
            self.singletask_tokenizer = None
        
        if keep_model != 'multitask_2' and self.multitask_2_model is not None:
            logger.info("Unloading Multi-Task-2 model to free memory")

            # Delete references to allow garbage collection
            del self.multitask_2_model
            del self.multitask_2_tokenizer

            # Set attributes to None 
            self.multitask_2_model = None
            self.multitask_2_tokenizer = None
        
        if keep_model != 'multitask_4' and self.multitask_4_model is not None:
            logger.info("Unloading Multi-Task-4 model to free memory")

            # Delete references to allow garbage collection
            del self.multitask_4_model
            del self.multitask_4_tokenizer

            # Set attributes to None 
            self.multitask_4_model = None
            self.multitask_4_tokenizer = None
        
        # Force garbage collection
        gc.collect()
        logger.info("Memory freed")
    
    def load_model_on_demand(self, model_name: str):
        """Load a specific model on demand, unloading others to save memory"""
        logger.info(f"Ensuring {model_name} is loaded...")
        
        # LogReg is always loaded (limited memory)
        if model_name == 'logreg':
            if self.logreg_model is None:
                self.load_logreg()
            return
        
        # For transformers, unload others and load requested one
        if model_name == 'singletask':
            if self.singletask_model is None:
                logger.info("Loading Single-Task model (this may take 20-30 seconds)...")

                # Unload other models
                self.unload_transformers(keep_model='singletask')

                # Load singletask
                self.load_singletask()

                # Update currently loaded
                self.currently_loaded_transformer = 'singletask'
                logger.info("Single-Task model loaded and ready")
            else:
                logger.info("Single-Task model already loaded")
        
        elif model_name == 'multitask_2':
            if self.multitask_2_model is None:
                logger.info("Loading Multi-Task-2 model (this may take 20-30 seconds)...")

                # Unload other models
                self.unload_transformers(keep_model='multitask_2')

                # Load multitask (2 heads)
                self.load_multitask_2()

                # Update currently loaded
                self.currently_loaded_transformer = 'multitask_2'
                logger.info("Multi-Task-2 model loaded and ready")
            else:
                logger.info("Multi-Task-2 model already loaded")
        
        elif model_name == 'multitask_4':
            if self.multitask_4_model is None:
                logger.info("Loading Multi-Task-4 model (this may take 20-30 seconds)...")
                
                # Unload other models
                self.unload_transformers(keep_model='multitask_4')

                # Load multitask (4 heads)
                self.load_multitask_4()

                # Update currently loaded
                self.currently_loaded_transformer = 'multitask_4'
                logger.info("Multi-Task-4 model loaded and ready")
            else:
                logger.info("Multi-Task-4 model already loaded")
    
    def load_logreg(self):
        """Load logistic regression model and vectorizer"""
        try:
            logger.info("Loading Logistic Regression model...")
            
            # Load model
            model_path = self.models_dir / "logistic_regression_model.pkl"
            with open(model_path, 'rb') as f:
                self.logreg_model = pickle.load(f)
            
            # Load vectorizer
            vectorizer_path = self.models_dir / "tfidf_vectorizer.pkl"
            with open(vectorizer_path, 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)
            
            logger.info("Loaded Logistic Regression model")
            
        except Exception as e:
            logger.error(f"Failed to load LogReg model: {e}")
            raise
    
    def load_singletask(self):
        """Load single-task RoBERTa model"""
        try:
            model_path = self.models_dir / "best_singletask.pt"
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize tokenizer
            self.singletask_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            
            # Initialize model
            self.singletask_model = RobertaForSequenceClassification.from_pretrained(
                'roberta-base',
                num_labels=2
            )
            
            # Load weights (handle different checkpoint formats)
            if 'model_state_dict' in checkpoint:
                # Standard format
                self.singletask_model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                # Alternative format
                self.singletask_model.load_state_dict(checkpoint['state_dict'])
            else:
                # Checkpoint might be the state dict itself
                self.singletask_model.load_state_dict(checkpoint)
            
            self.singletask_model.to(self.device)
            self.singletask_model.eval()
            
            logger.info("Loaded Single-Task model")
            
        except Exception as e:
            logger.error(f"Failed to load Single-Task model: {e}")
            raise
    
    def load_multitask_2(self):
        """Load 2-head multi-task RoBERTa model"""
        try:
            model_path = self.models_dir / "best_multitask_2.pt"
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize tokenizer
            self.multitask_2_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            
            # Initialize model
            self.multitask_2_model = MultiTaskRoBERTa(tasks=['Q_overall', 'Q2_harmful'])
            
            # Load weights
            self.multitask_2_model.load_state_dict(checkpoint['model_state_dict'])
            self.multitask_2_model.to(self.device)
            self.multitask_2_model.eval()
            
            logger.info("Loaded Multi-Task-2 model")
            
        except Exception as e:
            logger.error(f"Failed to load Multi-Task-2 model: {e}")
            raise
    
    def load_multitask_4(self):
        """Load 4-head multi-task RoBERTa model"""
        try:
            model_path = self.models_dir / "best_multitask_4.pt"
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Initialize tokenizer
            self.multitask_4_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            
            # Initialize model
            self.multitask_4_model = MultiTaskRoBERTa(tasks=['Q_overall', 'Q2_harmful', 'Q3_bias', 'Q6_policy'])
            
            # Load weights
            self.multitask_4_model.load_state_dict(checkpoint['model_state_dict'])
            self.multitask_4_model.to(self.device)
            self.multitask_4_model.eval()
            
            logger.info("Loaded Multi-Task-4 model")
            
        except Exception as e:
            logger.error(f"Failed to load Multi-Task-4 model: {e}")
            raise

    def predict_logreg(self, text: str) -> dict:
        """Make prediction using logistic regression"""
        if self.logreg_model is None:
            raise ValueError("LogReg model not loaded")
        
        # Vectorize text
        X = self.tfidf_vectorizer.transform([text])
        
        # Get prediction and probability
        prediction = self.logreg_model.predict(X)[0]
        probability = self.logreg_model.predict_proba(X)[0]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability[1]),  # Probability of unsafe class
            "label": "Unsafe" if prediction == 1 else "Safe"
        }
    
    def predict_singletask(self, text: str) -> dict:
        """Make prediction using single-task model"""
        if self.singletask_model is None:
            raise ValueError("Single-task model not loaded")
        
        # Tokenize
        inputs = self.singletask_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = self.singletask_model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
            prediction = torch.argmax(logits, dim=-1).item()
            probability = probs[0][1].item()  # Probability of unsafe class
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "label": "Unsafe" if prediction == 1 else "Safe"
        }
    
    def predict_multitask(self, text: str, task: str, num_heads: int) -> dict:
        """Make prediction using multi-task model"""
        
        # Select correct model
        if num_heads == 2:
            model = self.multitask_2_model
            tokenizer = self.multitask_2_tokenizer
        elif num_heads == 4:
            model = self.multitask_4_model
            tokenizer = self.multitask_4_tokenizer
        else:
            raise ValueError(f"Invalid num_heads: {num_heads}")
        
        if model is None:
            raise ValueError(f"Multi-task {num_heads} model not loaded")
        
        # Validate task
        if task not in ['Q_overall', 'Q2_harmful', 'Q3_bias', 'Q6_policy']:
            raise ValueError(f"Invalid task: {task}")
        
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs) 
            logit = outputs[task]  
            
            # Single output with sigmoid (BCEWithLogitsLoss style)
            probability = torch.sigmoid(logit).squeeze().item() 
            prediction = 1 if probability > 0.5 else 0
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "label": "Unsafe" if prediction == 1 else "Safe"
        }