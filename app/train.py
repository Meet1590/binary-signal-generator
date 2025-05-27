import os
import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Import local modules
from data_processor import (
    process_data_for_training,
    create_train_val_test_split,
    create_windowed_dataset
)

# Import models
from models import (
    RandomForestModel,
    GradientBoostingModel,
    XGBoostModel,
    LightGBMModel,
    SVMModel,
    LogisticRegressionModel,
    LSTMModel,
    GRUModel,
    CNNLSTMModel,
    ARIMAModel,
    AutoARIMAModel,
    EnsembleModel
)

def plot_confusion_matrix(cm, class_names, title='Confusion Matrix', figsize=(10, 8)):
    """
    Plot a confusion matrix
    """
    plt.figure(figsize=figsize)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add class labels
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add values to the plot
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    return plt

def evaluate_model(model, X, y_true, class_names=None):
    """
    Evaluate a model and print metrics
    
    Args:
        model: Trained model
        X: Features
        y_true: True labels
        class_names: List of class names
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Make predictions
    y_pred = model.predict(X)
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Print results
    print(f"Model: {model.name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    # Plot confusion matrix if class names are provided
    if class_names:
        plt = plot_confusion_matrix(cm, class_names, title=f'{model.name} Confusion Matrix')
        plt.savefig(f"results/{model.name}_confusion_matrix.png")
        plt.close()
    
    return metrics

def create_model_factory():
    """
    Create a factory of all available models
    
    Returns:
        Dictionary mapping model names to functions that create the model
    """
    return {
        'random_forest': lambda: RandomForestModel(n_estimators=200, max_depth=15),
        'gradient_boosting': lambda: GradientBoostingModel(n_estimators=200, learning_rate=0.05, max_depth=5),
        'xgboost': lambda: XGBoostModel(n_estimators=200, learning_rate=0.05, max_depth=5),
        'lightgbm': lambda: LightGBMModel(n_estimators=200, learning_rate=0.05, max_depth=5),
        'svm': lambda: SVMModel(C=1.0, kernel='rbf'),
        'logistic_regression': lambda: LogisticRegressionModel(C=1.0),
        'lstm': lambda input_size, seq_length: LSTMModel(
            input_size=input_size,
            seq_length=seq_length,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            epochs=50
        ),
        'gru': lambda input_size, seq_length: GRUModel(
            input_size=input_size,
            seq_length=seq_length,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            epochs=50
        ),
        'cnn_lstm': lambda input_size, seq_length: CNNLSTMModel(
            input_size=input_size,
            seq_length=seq_length,
            hidden_size=128,
            num_layers=2,
            dropout=0.3,
            epochs=50
        ),
        'arima': lambda: ARIMAModel(order=(5,1,0)),
        'auto_arima': lambda: AutoARIMAModel(
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            seasonal=False
        )
    }

def create_ensemble(models, weights=None):
    """
    Create an ensemble model from a list of models
    
    Args:
        models: List of trained models
        weights: Optional list of weights for each model
        
    Returns:
        Trained ensemble model
    """
    ensemble = EnsembleModel()
    
    for i, model in enumerate(models):
        weight = weights[i] if weights else 1.0
        ensemble.add_model(model, weight)
    
    return ensemble

def save_evaluation_results(results, filepath):
    """
    Save evaluation results to a JSON file
    
    Args:
        results: Dictionary of evaluation results
        filepath: Path to save the results
    """
    # Convert any numpy values to Python types
    cleaned_results = {}
    for model_name, metrics in results.items():
        cleaned_results[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1'])
        }
    
    # Save to file
    with open(filepath, 'w') as f:
        json.dump(cleaned_results, f, indent=4)
    
    print(f"Evaluation results saved to {filepath}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train and evaluate forex signal prediction models')
    
    parser.add_argument('--data_file', type=str, required=True,
                        help='Path to the input data file')
    
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    
    parser.add_argument('--models', type=str, nargs='+', 
                        default=['random_forest', 'lightgbm', 'lstm', 'gru'],
                        help='List of models to train')
    
    parser.add_argument('--window_size', type=int, default=10,
                        help='Window size for sequence models')
    
    parser.add_argument('--create_ensemble', action='store_true',
                        help='Create an ensemble of all trained models')
    
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit number of rows to process (for testing)')
    
    parser.add_argument('--lookahead', type=int, default=1,
                        help='Number of periods to look ahead for labeling')
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'models'), exist_ok=True)
    
    # Process data
    print("\n=== Processing Data ===")
    data = process_data_for_training(
        args.data_file,
        output_dir=args.output_dir,
        lookahead=args.lookahead,
        limit=args.limit
    )
    
    # Split data
    print("\n=== Splitting Data ===")
    train_df, val_df, test_df = create_train_val_test_split(data)
    
    # Create windowed datasets
    print("\n=== Creating Windowed Datasets ===")
    X_train, y_train = create_windowed_dataset(train_df, window_size=args.window_size, batch_size=1000)
    X_val, y_val = create_windowed_dataset(val_df, window_size=args.window_size, batch_size=1000)
    X_test, y_test = create_windowed_dataset(test_df, window_size=args.window_size, batch_size=1000)
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Get unique classes and their names
    unique_classes = np.unique(y_train)
    class_names = ['NEUTRAL', 'UP', 'DOWN']
    print(f"Classes: {unique_classes}")
    
    # Get model factory
    model_factory = create_model_factory()
    
    # Train and evaluate models
    print("\n=== Training and Evaluating Models ===")
    trained_models = []
    evaluation_results = {}
    
    for model_name in args.models:
        print(f"\nTraining {model_name}...")
        
        # Create model
        if model_name in ['lstm', 'gru', 'cnn_lstm']:
            # For deep learning models that need input_size and seq_length
            input_size = X_train.shape[2]
            seq_length = X_train.shape[1]
            model = model_factory[model_name](input_size, seq_length)
        else:
            # For other models
            model = model_factory[model_name]()
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Evaluate on validation set
            metrics = evaluate_model(model, X_val, y_val, class_names)
            evaluation_results[model_name] = metrics
            
            # Save model
            model_path = os.path.join(args.output_dir, 'models', f"{model_name}_model.pkl")
            model.save(model_path)
            
            # Add to list of trained models
            trained_models.append(model)
            
        except Exception as e:
            print(f"Error training {model_name}: {str(e)}")
    
    # Create and evaluate ensemble if requested
    if args.create_ensemble and len(trained_models) > 1:
        print("\nCreating ensemble model...")
        
        # Create ensemble with weights based on validation accuracy
        weights = [evaluation_results[model.name]['accuracy'] for model in trained_models]
        ensemble = create_ensemble(trained_models, weights)
        
        # Evaluate ensemble
        ensemble_metrics = evaluate_model(ensemble, X_val, y_val, class_names)
        evaluation_results['ensemble'] = ensemble_metrics
        
        # Save ensemble model
        ensemble_path = os.path.join(args.output_dir, 'models', "ensemble_model.pkl")
        ensemble.save(ensemble_path)
        
        # Add to list of trained models
        trained_models.append(ensemble)
    
    # Find best model based on validation accuracy
    best_model_name = max(evaluation_results, key=lambda k: evaluation_results[k]['accuracy'])
    best_model = next(model for model in trained_models if model.name == best_model_name)
    
    # Evaluate best model on test set
    print(f"\n=== Evaluating Best Model ({best_model_name}) on Test Set ===")
    test_metrics = evaluate_model(best_model, X_test, y_test, class_names)
    
    # Save evaluation results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    save_evaluation_results(evaluation_results, results_path)
    
    # Generate sample prediction with best model
    print("\n=== Sample Prediction ===")
    sample_data = X_test[:1]  # Take first sample from test set
    signal = best_model.generate_signal(sample_data)
    print(json.dumps(signal, indent=2))
    
    # Save best model as "final_model.pkl"
    final_model_path = os.path.join(args.output_dir, 'models', "final_model.pkl")
    best_model.save(final_model_path)
    print(f"\nBest model ({best_model_name}) saved as {final_model_path}")

if __name__ == "__main__":
    main() 