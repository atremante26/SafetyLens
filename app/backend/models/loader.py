import torch
import pickle
from pathlib import Path
from transformers import RobertaForSequenceClassification, RobertaTokenizer
import logging

logger = logging.getLogger(__name__)

# Import MultiTaskRoBERTa from project root
import importlib.util

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
multi_task_path = PROJECT_ROOT / "models" / "multi_task_transformer.py"

spec = importlib.util.spec_from_file_location("multi_task_transformer", multi_task_path)
multi_task_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(multi_task_module)

MultiTaskRoBERTa = multi_task_module.MultiTaskRoBERTa
load_model = multi_task_module.load_model


class ModelLoader:
    """Handles loading and management of all models with lazy loading support"""
    
    def __init__(self):
        self.device = torch.device('cpu')
        logger.info(f"Using device: {self.device}")
        
        # Model paths
        self.models_dir = Path(__file__).parent.parent.parent / "models" / "checkpoints"
        
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
        import gc
        
        if keep_model != 'singletask' and self.singletask_model is not None:
            logger.info("Unloading Single-Task model to free memory")
            del self.singletask_model
            del self.singletask_tokenizer
            self.singletask_model = None
            self.singletask_tokenizer = None
        
        if keep_model != 'multitask_2' and self.multitask_2_model is not None:
            logger.info("Unloading Multi-Task-2 model to free memory")
            del self.multitask_2_model
            del self.multitask_2_tokenizer
            self.multitask_2_model = None
            self.multitask_2_tokenizer = None
        
        if keep_model != 'multitask_4' and self.multitask_4_model is not None:
            logger.info("Unloading Multi-Task-4 model to free memory")
            del self.multitask_4_model
            del self.multitask_4_tokenizer
            self.multitask_4_model = None
            self.multitask_4_tokenizer = None
        
        # Force garbage collection
        gc.collect()
        logger.info("Memory freed")
    
    def load_model_on_demand(self, model_name: str):
        """Load a specific model on demand, unloading others to save memory"""
        logger.info(f"Ensuring {model_name} is loaded...")
        
        # LogReg is always loaded (tiny memory footprint)
        if model_name == 'logreg':
            if self.logreg_model is None:
                self.load_logreg()
            return
        
        # For transformers, unload others and load requested one
        if model_name == 'singletask':
            if self.singletask_model is None:
                logger.info("Loading Single-Task model (this may take 20-30 seconds)...")
                self.unload_transformers(keep_model='singletask')
                self.load_singletask()
                self.currently_loaded_transformer = 'singletask'
                logger.info("Single-Task model loaded and ready")
            else:
                logger.info("Single-Task model already loaded")
        
        elif model_name == 'multitask_2':
            if self.multitask_2_model is None:
                logger.info("Loading Multi-Task-2 model (this may take 20-30 seconds)...")
                self.unload_transformers(keep_model='multitask_2')
                self.load_multitask_2()
                self.currently_loaded_transformer = 'multitask_2'
                logger.info("Multi-Task-2 model loaded and ready")
            else:
                logger.info("Multi-Task-2 model already loaded")
        
        elif model_name == 'multitask_4':
            if self.multitask_4_model is None:
                logger.info("Loading Multi-Task-4 model (this may take 20-30 seconds)...")
                self.unload_transformers(keep_model='multitask_4')
                self.load_multitask_4()
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
            
            # Load weights
            self.singletask_model.load_state_dict(checkpoint['model_state_dict'])
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
            self.multitask_2_model = MultiTaskRoBERTa(num_tasks=2)
            
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
            self.multitask_4_model = MultiTaskRoBERTa(num_tasks=4)
            
            # Load weights
            self.multitask_4_model.load_state_dict(checkpoint['model_state_dict'])
            self.multitask_4_model.to(self.device)
            self.multitask_4_model.eval()
            
            logger.info("Loaded Multi-Task-4 model")
            
        except Exception as e:
            logger.error(f"Failed to load Multi-Task-4 model: {e}")
            raise