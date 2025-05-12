#!/usr/bin/env python3
"""
Script to create dummy ML models from social media dataset.

This script loads a dataset, preprocesses it, and creates several dummy models
for predicting academic performance, mental health, sleep patterns, and addiction levels.
"""

import joblib
import logging
import numpy as np
import os
import pandas as pd
import sys
from dataclasses import dataclass
from pathlib import Path
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Optional, Tuple, Union

# Set up project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOGS_DIR / 'model_creation.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Constants and configuration
@dataclass
class ModelConfig:
    """Configuration for model creation."""
    name: str
    target_columns: List[str]
    model_type: str  # 'classifier' or 'regressor'
    strategy: str
    output_filename: str

# Model configurations
MODEL_CONFIGS = [
    ModelConfig(
        name="academic_performance",
        target_columns=[
            'Affects_Academic_Performance', 'affects_academic_performance',
            'Academic_Performance_Impact', 'academic_impact'
        ],
        model_type="classifier",
        strategy="most_frequent",
        output_filename="academic_performance_model.joblib"
    ),
    ModelConfig(
        name="mental_health",
        target_columns=[
            'Mental_Health_Score', 'mental_health_score',
            'Mental_Health', 'mental_score'
        ],
        model_type="regressor",
        strategy="mean",
        output_filename="mental_health_model.joblib"
    ),
    ModelConfig(
        name="sleep",
        target_columns=[
            'Sleep_Hours_Per_Night', 'sleep_hours_per_night',
            'Sleep_Hours', 'sleep_duration'
        ],
        model_type="regressor",
        strategy="mean",
        output_filename="sleep_model.joblib"
    ),
    ModelConfig(
        name="addiction",
        target_columns=[
            'Addicted_Score', 'addicted_score',
            'Addiction_Score', 'addiction_level'
        ],
        model_type="classifier",
        strategy="most_frequent",
        output_filename="addiction_model.joblib"
    )
]

# Feature candidates
POSSIBLE_FEATURES = [
    'Age', 'age',
    'Avg_Daily_Usage_Hours', 'avg_daily_usage_hours', 'daily_usage_hours',
    'Sleep_Hours_Per_Night', 'sleep_hours_per_night', 'sleep_hours',
    'Conflicts_Over_Social_Media', 'conflicts_over_social_media', 'conflicts'
]


class ModelCreator:
    """Class to handle the creation of ML models from dataset."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize the ModelCreator.
        
        Args:
            project_root: Path to the project root directory. If None, uses the parent directory of this script.
        """
        self.project_root = project_root or PROJECT_ROOT
        self.models_dir = self.project_root / "data" / "models"
        self.raw_data_dir = self.project_root / "data" / "raw"
        
        # Ensure directories exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Change to project root directory
        os.chdir(self.project_root)
        logger.info(f"Working directory: {os.getcwd()}")
    
    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset from the raw data directory.
        
        Returns:
            DataFrame containing the loaded dataset.
            
        Raises:
            FileNotFoundError: If no CSV files are found in the raw data directory.
        """
        logger.info("Loading dataset from raw data directory...")
        
        csv_files = list(self.raw_data_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.raw_data_dir}")
        
        logger.info(f"Found CSV files: {[f.name for f in csv_files]}")
        
        # Use the first CSV file or allow specification
        data_file = csv_files[0]
        logger.info(f"Loading {data_file.name}...")
        
        try:
            df = pd.read_csv(data_file)
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"Columns: {df.columns.tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the dataset.
        
        Args:
            df: Raw DataFrame to preprocess.
            
        Returns:
            Preprocessed DataFrame.
        """
        logger.info("Preprocessing data...")
        
        # Create a copy to avoid modifying original
        df_processed = df.copy()
        
        # Check for missing values
        missing = df_processed.isnull().sum()
        if missing.any():
            logger.warning("Missing values detected:")
            for col, count in missing[missing > 0].items():
                logger.warning(f"  - {col}: {count} missing values")
        
        # Handle categorical variables
        categorical_cols = df_processed.select_dtypes(include=['object']).columns
        label_encoders = {}
        
        for col in categorical_cols:
            if col in df_processed.columns:
                le = LabelEncoder()
                df_processed[col] = le.fit_transform(df_processed[col].astype(str))
                label_encoders[col] = le
        
        # Save label encoders for later use
        joblib.dump(label_encoders, self.models_dir / 'label_encoders.joblib')
        logger.info("Saved label encoders")
        
        return df_processed
    
    def identify_features(self, df: pd.DataFrame) -> List[str]:
        """Identify feature columns from the dataset.
        
        Args:
            df: DataFrame to analyze.
            
        Returns:
            List of feature column names.
        """
        # Find which features exist in the dataset
        actual_features = [feature for feature in POSSIBLE_FEATURES if feature in df.columns]
        
        logger.info(f"Identified features: {actual_features}")
        
        if not actual_features:
            logger.warning("No standard features found. Using first 4 numeric columns.")
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            actual_features = numeric_cols[:4].tolist()
            logger.info(f"Using numeric features: {actual_features}")
        
        return actual_features
    
    def create_model(
        self, 
        df: pd.DataFrame, 
        features: List[str], 
        config: ModelConfig
    ) -> Tuple[Optional[str], Optional[float]]:
        """Create and save a model based on the provided configuration.
        
        Args:
            df: Preprocessed DataFrame.
            features: List of feature column names.
            config: Model configuration.
            
        Returns:
            Tuple of (target_column, model_score) if successful, (None, None) otherwise.
        """
        # Find the target column
        target_col = None
        for col in config.target_columns:
            if col in df.columns:
                target_col = col
                break
        
        if not target_col:
            logger.warning(f"No target column found for {config.name} model. Skipping.")
            return None, None
        
        logger.info(f"Creating {config.name} model with target: {target_col}")
        
        # Prepare feature matrix and target
        X = df[features]
        y = df[target_col]
        
        # Log target information
        if config.model_type == "classifier":
            logger.info(f"Target values: {y.value_counts().to_dict()}")
        else:
            logger.info(f"Target range: {y.min()} - {y.max()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create and train model
        if config.model_type == "classifier":
            model = DummyClassifier(strategy=config.strategy, random_state=42)
        else:
            model = DummyRegressor(strategy=config.strategy)
        
        model.fit(X_train, y_train)
        
        # Calculate and log score
        score = model.score(X_test, y_test)
        metric_name = "accuracy" if config.model_type == "classifier" else "R² score"
        logger.info(f"Model {metric_name}: {score:.3f}")
        
        # Save model
        joblib.dump(model, self.models_dir / config.output_filename)
        logger.info(f"Saved {config.name} model")
        
        return target_col, score
    
    def create_all_models(self, df: pd.DataFrame) -> Dict[str, Dict]:
        """Create all models defined in MODEL_CONFIGS.
        
        Args:
            df: Preprocessed DataFrame.
            
        Returns:
            Dictionary with information about created models.
        """
        logger.info("Creating models from dataset...")
        
        # Identify features
        features = self.identify_features(df)
        
        # Store model information
        model_info = {
            'features': features,
            'models': {}
        }
        
        # Create each model
        for config in MODEL_CONFIGS:
            target_col, score = self.create_model(df, features, config)
            
            if target_col:
                model_info['models'][config.name] = {
                    'target': target_col,
                    'score': score,
                    'model_type': config.model_type,
                    'file': config.output_filename
                }
        
        # Save feature and model information
        joblib.dump(model_info, self.models_dir / 'model_info.joblib')
        logger.info("Saved model information")
        
        return model_info
    
    def run(self) -> None:
        """Run the full model creation pipeline."""
        try:
            logger.info("🚀 Starting ML model creation process...")
            
            # Load dataset
            df = self.load_dataset()
            logger.info(f"Dataset shape: {df.shape}")
            
            # Preprocess data
            df_processed = self.preprocess_data(df)
            
            # Create models
            model_info = self.create_all_models(df_processed)
            
            # Log summary
            logger.info("✨ Model creation completed successfully!")
            logger.info("Models created:")
            for name, info in model_info['models'].items():
                logger.info(f"  - {name}: {info['file']} (score: {info['score']:.3f})")
            
        except Exception as e:
            logger.error(f"Error creating models: {e}", exc_info=True)
            raise


def main():
    """Main entry point."""
    try:
        creator = ModelCreator()
        creator.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()