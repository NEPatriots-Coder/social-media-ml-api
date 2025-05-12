import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from pathlib import Path
import os

# Make sure we're in the project root directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Create models directory
models_dir = Path("data/models")
models_dir.mkdir(parents=True, exist_ok=True)

def load_real_data():
    """Load the actual Kaggle dataset"""
    print("Loading real dataset from Kaggle...")
    
    # List CSV files in data/raw to find your file
    csv_files = list(Path("data/raw").glob("*.csv"))
    print(f"Found CSV files: {[f.name for f in csv_files]}")
    
    if not csv_files:
        raise FileNotFoundError("No CSV files found in data/raw/")
    
    # Use the first CSV file (or specify which one if you have multiple)
    data_file = csv_files[0]
    print(f"Loading {data_file.name}...")
    
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print("Columns:", df.columns.tolist())
    
    return df

def preprocess_data(df):
    """Basic preprocessing for the real dataset"""
    print("Preprocessing data...")
    
    # Create a copy to avoid modifying original
    df_processed = df.copy()
    
    # Handle categorical variables
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            label_encoders[col] = le
    
    # Save label encoders for later use
    joblib.dump(label_encoders, models_dir / 'label_encoders.joblib')
    print("✅ Saved label encoders")
    
    return df_processed

def create_models_from_real_data(df):
    """Create models using the actual dataset columns"""
    print("Creating models from real data...")
    
    # Define feature columns - adjust these based on your actual columns
    # These are common feature names from social media datasets
    possible_features = [
        'Age', 'age',
        'Avg_Daily_Usage_Hours', 'avg_daily_usage_hours', 'daily_usage_hours',
        'Sleep_Hours_Per_Night', 'sleep_hours_per_night', 'sleep_hours',
        'Conflicts_Over_Social_Media', 'conflicts_over_social_media', 'conflicts'
    ]
    
    # Find which features exist in your dataset
    actual_features = []
    for feature in possible_features:
        if feature in df.columns:
            actual_features.append(feature)
    
    print(f"Using features: {actual_features}")
    
    if not actual_features:
        print("Warning: No standard features found. Using first 4 numeric columns.")
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        actual_features = numeric_cols[:4].tolist()
        print(f"Using numeric features: {actual_features}")
    
    # Prepare feature matrix
    X = df[actual_features]
    
    # 1. Academic Performance Model
    academic_targets = [
        'Affects_Academic_Performance', 'affects_academic_performance',
        'Academic_Performance_Impact', 'academic_impact'
    ]
    
    academic_target = None
    for target in academic_targets:
        if target in df.columns:
            academic_target = target
            break
    
    if academic_target:
        y = df[academic_target]
        print(f"Creating academic performance model with target: {academic_target}")
        print(f"Target values: {y.value_counts().to_dict()}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create and train model
        model = DummyClassifier(strategy='most_frequent', random_state=42)
        model.fit(X_train, y_train)
        
        # Calculate accuracy
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.3f}")
        
        joblib.dump(model, models_dir / 'academic_performance_model.joblib')
        print("✅ Created academic performance model")
    
    # 2. Mental Health Model
    mental_health_targets = [
        'Mental_Health_Score', 'mental_health_score',
        'Mental_Health', 'mental_score'
    ]
    
    mental_target = None
    for target in mental_health_targets:
        if target in df.columns:
            mental_target = target
            break
    
    if mental_target:
        y = df[mental_target]
        print(f"Creating mental health model with target: {mental_target}")
        print(f"Target range: {y.min()} - {y.max()}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = DummyRegressor(strategy='mean')
        model.fit(X_train, y_train)
        
        # Calculate R2 score
        r2 = model.score(X_test, y_test)
        print(f"Model R2 score: {r2:.3f}")
        
        joblib.dump(model, models_dir / 'mental_health_model.joblib')
        print("✅ Created mental health model")
    
    # 3. Sleep Model
    sleep_features = actual_features[:3]  # Use first 3 features for sleep model
    sleep_targets = [
        'Sleep_Hours_Per_Night', 'sleep_hours_per_night',
        'Sleep_Hours', 'sleep_duration'
    ]
    
    sleep_target = None
    for target in sleep_targets:
        if target in df.columns:
            sleep_target = target
            break
    
    if sleep_target:
        X_sleep = df[sleep_features]
        y = df[sleep_target]
        print(f"Creating sleep model with target: {sleep_target}")
        print(f"Target range: {y.min()} - {y.max()}")
        
        X_train, X_test, y_train, y_test = train_test_split(X_sleep, y, test_size=0.2, random_state=42)
        
        model = DummyRegressor(strategy='mean')
        model.fit(X_train, y_train)
        
        r2 = model.score(X_test, y_test)
        print(f"Model R2 score: {r2:.3f}")
        
        joblib.dump(model, models_dir / 'sleep_model.joblib')
        print("✅ Created sleep model")
    
    # 4. Addiction Model
    addiction_targets = [
        'Addicted_Score', 'addicted_score',
        'Addiction_Score', 'addiction_level'
    ]
    
    addiction_target = None
    for target in addiction_targets:
        if target in df.columns:
            addiction_target = target
            break
    
    if addiction_target:
        y = df[addiction_target]
        print(f"Creating addiction model with target: {addiction_target}")
        print(f"Target range: {y.min()} - {y.max()}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # For addiction score, we'll treat it as classification
        model = DummyClassifier(strategy='most_frequent', random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"Model accuracy: {accuracy:.3f}")
        
        joblib.dump(model, models_dir / 'addiction_model.joblib')
        print("✅ Created addiction model")
    
    # Save feature information
    feature_info = {
        'features': actual_features,
        'academic_target': academic_target,
        'mental_health_target': mental_target,
        'sleep_target': sleep_target,
        'addiction_target': addiction_target
    }
    
    joblib.dump(feature_info, models_dir / 'feature_info.joblib')
    print("✅ Saved feature information")

def main():
    print("🚀 Creating ML models from your real dataset...")
    print(f"Working directory: {os.getcwd()}")
    
    # Load real data
    try:
        df = load_real_data()
        print(f"\nDataset shape: {df.shape}")
        print("\nFirst few rows:")
        print(df.head())
        
        # Check for missing values
        missing = df.isnull().sum()
        if missing.any():
            print(f"\nMissing values:")
            print(missing[missing > 0])
        
        # Preprocess data
        df_processed = preprocess_data(df)
        
        # Create models
        create_models_from_real_data(df_processed)
        
        print(f"\n✨ All models created successfully!")
        print("Models saved to:")
        for model_file in models_dir.glob("*.joblib"):
            print(f"  - {model_file}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
EOF
