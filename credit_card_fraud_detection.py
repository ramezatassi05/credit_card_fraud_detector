import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

def load_and_preprocess_data():
    """Load and preprocess the credit card fraud dataset."""
    print("Loading dataset...")
    df = pd.read_csv('creditcard.csv')
    
    # Check for missing values
    print("\nChecking for missing values:")
    print(df.isnull().sum())
    
    # Basic statistics
    print("\nBasic statistics:")
    print(df.describe())
    
    # Class distribution
    print("\nClass distribution:")
    print(df['Class'].value_counts(normalize=True))
    
    return df

def feature_engineering(df):
    """Perform feature engineering on the dataset."""
    # Create time-based features
    df['hour'] = df['Time'].apply(lambda x: np.floor(x / 3600) % 24)
    df['day'] = df['Time'].apply(lambda x: np.floor(x / (3600 * 24)) % 7)
    
    # Create interaction features
    for i in range(1, 29):
        for j in range(i+1, 29):
            df[f'V{i}_V{j}_interaction'] = df[f'V{i}'] * df[f'V{j}']
    
    return df

def prepare_data(df):
    """Prepare data for modeling."""
    # Separate features and target
    X = df.drop(['Time', 'Class'], axis=1)
    y = df['Class']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def handle_imbalance(X_train, y_train):
    """Handle class imbalance using SMOTE."""
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    return X_train_balanced, y_train_balanced

def train_models(X_train, y_train):
    """Train multiple models and return their predictions."""
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(random_state=42),
        'LightGBM': lgb.LGBMClassifier(random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def evaluate_models(models, X_test, y_test):
    """Evaluate all models and print their performance metrics."""
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        print(f"\n{name} Performance:")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

def plot_feature_importance(model, feature_names):
    """Plot feature importance for a given model."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        plt.figure(figsize=(12, 6))
        plt.title('Feature Importances')
        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.tight_layout()
        plt.show()

def main():
    # Load and preprocess data
    df = load_and_preprocess_data()
    
    # Feature engineering
    print("\nPerforming feature engineering...")
    df = feature_engineering(df)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(df)
    
    # Handle class imbalance
    print("\nHandling class imbalance...")
    X_train_balanced, y_train_balanced = handle_imbalance(X_train, y_train)
    
    # Train models
    trained_models = train_models(X_train_balanced, y_train_balanced)
    
    # Evaluate models
    evaluate_models(trained_models, X_test, y_test)
    
    # Plot feature importance for Random Forest
    feature_names = df.drop(['Time', 'Class'], axis=1).columns
    plot_feature_importance(trained_models['Random Forest'], feature_names)

if __name__ == "__main__":
    main()
