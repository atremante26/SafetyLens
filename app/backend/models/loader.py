import torch
import pickle
from pathlib import Path
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import logging

logger = logging.getLogger(__name__)

# Import MultiTaskRoBERTa from project root
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
multi_task_path = PROJECT_ROOT / "models" / "multi_task_transformer.py"

spec = importlib.util.spec_from_file_location("multi_task_transformer", multi_task_path)
multi_task_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_task_module)

MultiTaskRoBERTa = multi_task_module.MultiTaskRoBERTa
load_model = multi_task_module.load_model


class ModelLoader:
    """
    Centralized model loader for all SafetyLens models
    Handles loading and prediction for LogReg, Single-Task, and Multi-Task models
    """
    def __init__(self):
        """Initialize model loader"""
        self.logreg_model = None
        self.vectorizer = None
        self.singletask_model = None
        self.singletask_tokenizer = None
        self.multitask_2_model = None
        self.multitask_2_tokenizer = None
        self.multitask_4_model = None
        self.multitask_4_tokenizer = None
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def load_all_models(self):
        """Load all models into memory"""
        self.load_logreg()
        self.load_singletask()
        self.load_multitask_2()
        self.load_multitask_4()
    
    def load_logreg(self):
        """Load logistic regression model and TF-IDF vectorizer"""
        try:
            model_path = PROJECT_ROOT / "models" / "checkpoints" / "logistic_regression_model.pkl"
            vectorizer_path = PROJECT_ROOT / "models" / "checkpoints" / "tfidf_vectorizer.pkl"
            
            with open(model_path, 'rb') as f:
                self.logreg_model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            logger.info("Loaded Logistic Regression model")
        except Exception as e:
            logger.error(f"Failed to load LogReg model: {e}")
            raise
    
    def load_singletask(self):
        """Load single-task RoBERTa model"""
        try:
            model_path = PROJECT_ROOT / "models" / "checkpoints" / "best_singletask.pt"
            
            checkpoint = torch.load(model_path, map_location=self.device)
            
            self.singletask_model = RobertaForSequenceClassification.from_pretrained(
                "roberta-base",
                num_labels=2
            )
            
            self.singletask_model.load_state_dict(checkpoint, strict=False)
            self.singletask_model.to(self.device)
            self.singletask_model.eval()
            
            self.singletask_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            
            logger.info("Loaded Single-Task RoBERTa model")
        except Exception as e:
            logger.error(f"Failed to load Single-Task model: {e}")
            raise
    
    def load_multitask_2(self):
        """Load 2-head multi-task model"""
        try:
            model_path = PROJECT_ROOT / "models" / "checkpoints" / "best_multitask_2.pt"
            tasks = ['Q_overall', 'Q2_harmful']
            
            self.multitask_2_model, self.multitask_2_tokenizer = load_model(
                model_path, 
                self.device
            )
            
            logger.info(f"Loaded Multi-Task-2 model (tasks: {tasks})")
        except Exception as e:
            logger.error(f"Failed to load Multi-Task-2 model: {e}")
            raise
    
    def load_multitask_4(self):
        """Load 4-head multi-task model"""
        try:
            model_path = PROJECT_ROOT / "models" / "checkpoints" / "best_multitask_4.pt"
            tasks = ['Q_overall', 'Q2_harmful', 'Q3_bias', 'Q6_policy']
            
            self.multitask_4_model, self.multitask_4_tokenizer = load_model(
                model_path, 
                self.device
            )
            
            logger.info(f"Loaded Multi-Task-4 model (tasks: {tasks})")
        except Exception as e:
            logger.error(f"Failed to load Multi-Task-4 model: {e}")
            raise
    
    def predict_logreg(self, text: str) -> dict:
        """
        Predict using logistic regression model
        """
        X = self.vectorizer.transform([text])
        prediction = self.logreg_model.predict(X)[0]
        probability = self.logreg_model.predict_proba(X)[0][1]
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "label": "Unsafe" if prediction == 1 else "Safe"
        }
    
    def predict_singletask(self, text: str) -> dict:
        """
        Predict using single-task RoBERTa model
        """
        inputs = self.singletask_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.singletask_model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            probability = probs[0][1].item()
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "label": "Unsafe" if prediction == 1 else "Safe"
        }
    
    def predict_multitask(self, text: str, task: str = 'Q_overall', num_heads: int = 2) -> dict:
        """
        Predict using multi-task model
        """
        model = self.multitask_2_model if num_heads == 2 else self.multitask_4_model
        tokenizer = self.multitask_2_tokenizer if num_heads == 2 else self.multitask_4_tokenizer
        
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding='max_length'
        ).to(self.device)
        
        with torch.no_grad():
            logits_dict = model(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask']
            )
            
            logit = logits_dict[task]
            probability = torch.sigmoid(logit).item()
            prediction = 1 if probability > 0.5 else 0
        
        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "label": "Unsafe" if prediction == 1 else "Safe",
            "task": task
        }