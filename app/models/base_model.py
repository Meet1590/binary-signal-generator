from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime

class BaseSignalModel(ABC):
    """
    Abstract base class for all signal prediction models
    """
    def __init__(self, name="BaseModel"):
        self.name = name
        self.is_fitted = False
        self.model = None
    
    @abstractmethod
    def fit(self, X, y):
        """
        Train the model on the given data
        
        Args:
            X: Features array or DataFrame
            y: Target array or Series
        """
        pass
    
    @abstractmethod
    def predict(self, X):
        """
        Make binary predictions (1=UP, -1=DOWN, 0=NEUTRAL)
        
        Args:
            X: Features array or DataFrame
            
        Returns:
            Array of predictions
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X):
        """
        Predict class probabilities
        
        Args:
            X: Features array or DataFrame
            
        Returns:
            Array of class probabilities
        """
        pass
    
    def generate_signal(self, X, threshold=0.55):
        """
        Generate trading signal with confidence
        
        Args:
            X: Features array or DataFrame
            threshold: Probability threshold for signal generation
            
        Returns:
            Dictionary with signal information
        """
        # Get probabilities
        probas = self.predict_proba(X)
        
        # Get the prediction and confidence
        if len(probas.shape) == 2:  # Multi-class
            pred_class = np.argmax(probas, axis=1)[0]
            confidence = probas[0, pred_class]
            
            # Map class index to signal
            if pred_class == 0:
                signal = "NEUTRAL"
            elif pred_class == 1:
                signal = "UP"
            else:
                signal = "DOWN"
        else:  # Binary class
            # If probability is for positive class only
            confidence = probas[0]
            if confidence > threshold:
                signal = "UP"
            elif confidence < (1 - threshold):
                signal = "DOWN"
            else:
                signal = "NEUTRAL"
                confidence = 1 - abs(confidence - 0.5) * 2  # Adjust confidence for neutral
        
        # Current time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Prepare signal dict
        signal_dict = {
            "timestamp": current_time,
            "pair": "EUR/USD",
            "timeframe": "1m",
            "signal": signal,
            "confidence": round(float(confidence), 3),
            "model": self.name
        }
        
        return signal_dict
    
    def save(self, filepath):
        """
        Save the model to a file
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save the model
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        
        print(f"Model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load a model from a file
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"Model loaded from {filepath}")
        return model
    
    def evaluate(self, X, y_true):
        """
        Evaluate model performance
        
        Args:
            X: Features array or DataFrame
            y_true: True labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        # Make predictions
        y_pred = self.predict(X)
        
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
        print(f"Model: {self.name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        print(f"Confusion Matrix:\n{cm}")
        
        return metrics 