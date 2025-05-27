import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from .base_model import BaseSignalModel
from xgboost import XGBClassifier
import lightgbm as lgb

class RandomForestModel(BaseSignalModel):
    """
    Random Forest model for signal prediction
    """
    def __init__(self, n_estimators=100, max_depth=10, random_state=42):
        super().__init__(name="RandomForest")
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
    
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make binary predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.model.predict_proba(X)

class GradientBoostingModel(BaseSignalModel):
    """
    Gradient Boosting model for signal prediction
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        super().__init__(name="GradientBoosting")
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state
        )
    
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make binary predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.model.predict_proba(X)

class XGBoostModel(BaseSignalModel):
    """
    XGBoost model for signal prediction
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        super().__init__(name="XGBoost")
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            use_label_encoder=False,
            eval_metric='mlogloss'
        )
    
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make binary predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.model.predict_proba(X)

class LightGBMModel(BaseSignalModel):
    """
    LightGBM model for signal prediction
    """
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42):
        super().__init__(name="LightGBM")
        self.model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            class_weight='balanced',
            n_jobs=-1
        )
    
    def fit(self, X, y):
        """Train the model"""
        self.model.fit(X, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make binary predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Predict class probabilities"""
        return self.model.predict_proba(X)

class SVMModel(BaseSignalModel):
    """
    Support Vector Machine model for signal prediction
    """
    def __init__(self, C=1.0, kernel='rbf', probability=True, random_state=42):
        super().__init__(name="SVM")
        self.model = SVC(
            C=C,
            kernel=kernel,
            probability=probability,
            random_state=random_state,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        """Train the model with scaling"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make binary predictions with scaling"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities with scaling"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

class LogisticRegressionModel(BaseSignalModel):
    """
    Logistic Regression model for signal prediction
    """
    def __init__(self, C=1.0, random_state=42, max_iter=1000):
        super().__init__(name="LogisticRegression")
        self.model = LogisticRegression(
            C=C,
            random_state=random_state,
            max_iter=max_iter,
            class_weight='balanced',
            n_jobs=-1
        )
        self.scaler = StandardScaler()
    
    def fit(self, X, y):
        """Train the model with scaling"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make binary predictions with scaling"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X):
        """Predict class probabilities with scaling"""
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled) 