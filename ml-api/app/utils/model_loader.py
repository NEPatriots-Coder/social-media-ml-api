import joblib
import pickle
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
import json
from datetime import datetime

from app.core.config import get_settings

# Set up logging
logger = logging.getLogger(__name__)

# Global model manager instance
_model_manager: Optional['ModelManager'] = None


class ModelWrapper:
    """Wrapper class for ML models"""
    
    def __init__(self, model: BaseEstimator, model_type: str, version: str, preprocessor=None):
        self.model = model
        self.model_type = model_type
        self.version = version
        self.preprocessor = preprocessor
        self.loaded_at = datetime.utcnow()
        
    def predict(self, features: Dict[str, Any]) -> Tuple[Any, Any]:
        """Make prediction with the wrapped model"""
        try:
            # Convert to DataFrame
            df = pd.DataFrame([features])
            
            # Apply preprocessing if available
            if self.preprocessor:
                df = self.preprocessor.transform(df)
            
            # Make prediction based on model type
            if self.model_type == "classifier":
                prediction = self.model.predict(df)[0]
                probabilities = self.model.predict_proba(df)[0]
                
                # Convert to dictionary with class names
                classes = self.model.classes_
                prob_dict = {str(cls): float(prob) for cls, prob in zip(classes, probabilities)}
                
                return prediction, prob_dict
                
            elif self.model_type == "regressor":
                prediction = self.model.predict(df)[0]
                
                # Calculate prediction interval (simple approach)
                # In a real implementation, you'd use more sophisticated methods
                std_error = 0.1 * abs(prediction)  # Rough estimate
                interval = {
                    "lower": float(prediction - 1.96 * std_error),
                    "upper": float(prediction + 1.96 * std_error)
                }
                
                return float(prediction), interval
                
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
                
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


class ModelManager:
    """Manages loading and serving of ML models"""
    
    def __init__(self):
        self.models: Dict[str, ModelWrapper] = {}
        self.settings = get_settings()
        self.models_path = Path(self.settings.models_path)
        
    async def load_models(self):
        """Load all models from disk"""
        logger.info("Loading models...")
        
        # Define models to load
        model_configs = {
            "academic_performance_classifier": {
                "path": "academic_performance_model.joblib",
                "type": "classifier",
                "version": "1.0.0"
            },
            "mental_health_predictor": {
                "path": "mental_health_model.joblib",
                "type": "regressor",
                "version": "1.0.0"
            },
            "sleep_predictor": {
                "path": "sleep_model.joblib",
                "type": "regressor",
                "version": "1.0.0"
            },
            "addiction_classifier": {
                "path": "addiction_model.joblib",
                "type": "classifier",
                "version": "1.0.0"
            }
        }
        
        for model_name, config in model_configs.items():
            try:
                model_path = self.models_path / config["path"]
                
                # Check if model exists
                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    # Create a dummy model for testing
                    self._create_dummy_model(model_name, config)
                    continue
                
                # Load model
                model = joblib.load(model_path)
                
                # Check for preprocessor
                preprocessor_path = self.models_path / f"{model_name}_preprocessor.joblib"
                preprocessor = None
                if preprocessor_path.exists():
                    preprocessor = joblib.load(preprocessor_path)
                
                # Wrap the model
                wrapped_model = ModelWrapper(
                    model=model,
                    model_type=config["type"],
                    version=config["version"],
                    preprocessor=preprocessor
                )
                
                self.models[model_name] = wrapped_model
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                # Create a dummy model for development
                self._create_dummy_model(model_name, config)
    
    def _create_dummy_model(self, model_name: str, config: Dict[str, Any]):
        """Create a dummy model for testing when real models aren't available"""
        from sklearn.dummy import DummyClassifier, DummyRegressor
        
        logger.info(f"Creating dummy model for {model_name}")
        
        if config["type"] == "classifier":
            model = DummyClassifier(strategy="uniform")
            # Create dummy training data
            X_dummy = np.random.random((10, 5))
            y_dummy = np.random.choice(['Yes', 'No'], 10)
            model.fit(X_dummy, y_dummy)
        else:
            model = DummyRegressor(strategy="mean")
            # Create dummy training data
            X_dummy = np.random.random((10, 5))
            y_dummy = np.random.uniform(1, 10, 10)
            model.fit(X_dummy, y_dummy)
        
        wrapped_model = ModelWrapper(
            model=model,
            model_type=config["type"],
            version=f"{config['version']}-dummy",
            preprocessor=None
        )
        
        self.models[model_name] = wrapped_model
    
    def get_model(self, model_name: str) -> ModelWrapper:
        """Get a specific model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        return self.models[model_name]
    
    def are_models_loaded(self) -> bool:
        """Check if models are loaded"""
        return len(self.models) > 0
    
    def get_models_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        return {
            model_name: {
                "type": model.model_type,
                "version": model.version,
                "loaded_at": model.loaded_at.isoformat()
            }
            for model_name, model in self.models.items()
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for all models"""
        # This would typically load from a metrics file or database
        # For now, return dummy metrics
        return {
            "academic_performance_classifier": {
                "accuracy": 0.85,
                "precision": 0.83,
                "recall": 0.87,
                "f1_score": 0.85
            },
            "mental_health_predictor": {
                "mae": 0.8,
                "mse": 1.2,
                "r2_score": 0.75
            },
            "sleep_predictor": {
                "mae": 0.5,
                "mse": 0.8,
                "r2_score": 0.82
            },
            "addiction_classifier": {
                "accuracy": 0.78,
                "precision": 0.76,
                "recall": 0.80,
                "f1_score": 0.78
            }
        }


def get_model_manager() -> ModelManager:
    """Get the global model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
    return _model_manager