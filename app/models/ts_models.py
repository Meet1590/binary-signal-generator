import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from .base_model import BaseSignalModel
import warnings
import joblib

# Suppress warnings from statsmodels
warnings.filterwarnings("ignore")

class ARIMAModel(BaseSignalModel):
    """
    ARIMA model for time series prediction
    """
    def __init__(self, order=(5,1,0), seasonal_order=None, threshold=0.0):
        super().__init__(name="ARIMA")
        self.order = order
        self.seasonal_order = seasonal_order
        self.threshold = threshold
        self.model = None
    
    def fit(self, X, y=None):
        """
        Fit ARIMA model to the data
        
        Args:
            X: Time series data (array or DataFrame)
            y: Ignored (for compatibility with sklearn API)
        """
        # If X is 3D (windowed data), we need to extract the last value of each window
        if len(X.shape) == 3:
            # Take the close price from the last timestep of each window
            # Assuming close price is one of the features
            ts_data = pd.Series(X[:, -1, 0])  # Close price is typically the first feature
        elif isinstance(X, pd.DataFrame):
            # If it's a DataFrame, extract the close price
            ts_data = X['close'] if 'close' in X.columns else X.iloc[:, 0]
        else:
            # Assume X is already a 1D array or Series
            ts_data = X
        
        # Fit ARIMA model
        if self.seasonal_order:
            self.model = SARIMAX(
                ts_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
        else:
            self.model = ARIMA(
                ts_data,
                order=self.order
            ).fit()
        
        self.is_fitted = True
        return self
    
    def predict(self, X, steps=1):
        """
        Make point predictions
        
        Args:
            X: Time series data (array or DataFrame)
            steps: Number of steps to forecast
            
        Returns:
            Array of predictions (1=UP, -1=DOWN, 0=NEUTRAL)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        # Make forecasts
        forecast = self.model.forecast(steps=steps)
        
        # Extract the last known value
        if len(X.shape) == 3:
            last_value = X[-1, -1, 0]  # Last value of close price
        elif isinstance(X, pd.DataFrame):
            last_value = X['close'].iloc[-1] if 'close' in X.columns else X.iloc[-1, 0]
        else:
            last_value = X[-1]
        
        # Calculate predicted direction
        pct_change = (forecast[0] - last_value) / last_value * 100
        
        # Determine direction based on threshold
        if pct_change > self.threshold:
            return np.array([1])  # UP
        elif pct_change < -self.threshold:
            return np.array([-1])  # DOWN
        else:
            return np.array([0])  # NEUTRAL
    
    def predict_proba(self, X, steps=1):
        """
        Predict probabilities for each class
        
        Args:
            X: Time series data (array or DataFrame)
            steps: Number of steps to forecast
            
        Returns:
            Array of class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        # Make forecasts with confidence interval
        forecast = self.model.get_forecast(steps=steps)
        mean_forecast = forecast.predicted_mean[0]
        
        # Get confidence intervals
        conf_int = forecast.conf_int(alpha=0.05)
        lower_bound = conf_int.iloc[0, 0]
        upper_bound = conf_int.iloc[0, 1]
        
        # Extract the last known value
        if len(X.shape) == 3:
            last_value = X[-1, -1, 0]  # Last value of close price
        elif isinstance(X, pd.DataFrame):
            last_value = X['close'].iloc[-1] if 'close' in X.columns else X.iloc[-1, 0]
        else:
            last_value = X[-1]
        
        # Calculate predicted direction and confidence
        pct_change = (mean_forecast - last_value) / last_value * 100
        
        # Width of the confidence interval as a measure of uncertainty
        ci_width = upper_bound - lower_bound
        uncertainty = min(ci_width / (4 * abs(mean_forecast)), 0.5)
        
        # Calculate probabilities
        if pct_change > self.threshold:
            # UP with confidence proportional to pct_change and inversely to uncertainty
            up_prob = min(0.5 + abs(pct_change/10) - uncertainty, 0.95)
            return np.array([1 - up_prob, up_prob])
        elif pct_change < -self.threshold:
            # DOWN with confidence proportional to pct_change and inversely to uncertainty
            down_prob = min(0.5 + abs(pct_change/10) - uncertainty, 0.95)
            return np.array([down_prob, 1 - down_prob])
        else:
            # NEUTRAL with high uncertainty
            return np.array([0.5, 0.5])

class AutoARIMAModel(ARIMAModel):
    """
    Auto ARIMA model that automatically selects the best parameters
    """
    def __init__(self, start_p=1, start_q=1, max_p=5, max_q=5, 
                 seasonal=False, d=1, D=1, max_P=2, max_Q=2, m=7, threshold=0.0):
        super().__init__(threshold=threshold)
        self.name = "AutoARIMA"
        self.start_p = start_p
        self.start_q = start_q
        self.max_p = max_p
        self.max_q = max_q
        self.seasonal = seasonal
        self.d = d
        self.D = D
        self.max_P = max_P
        self.max_Q = max_Q
        self.m = m
    
    def fit(self, X, y=None):
        """
        Fit Auto ARIMA model to the data
        
        Args:
            X: Time series data (array or DataFrame)
            y: Ignored (for compatibility with sklearn API)
        """
        # If X is 3D (windowed data), we need to extract the last value of each window
        if len(X.shape) == 3:
            # Take the close price from the last timestep of each window
            ts_data = pd.Series(X[:, -1, 0])  # Close price is typically the first feature
        elif isinstance(X, pd.DataFrame):
            # If it's a DataFrame, extract the close price
            ts_data = X['close'] if 'close' in X.columns else X.iloc[:, 0]
        else:
            # Assume X is already a 1D array or Series
            ts_data = X
        
        # Use auto_arima to find the best parameters
        print("Finding optimal ARIMA parameters...")
        auto_model = auto_arima(
            ts_data,
            start_p=self.start_p,
            start_q=self.start_q,
            max_p=self.max_p,
            max_q=self.max_q,
            d=self.d,
            seasonal=self.seasonal,
            D=self.D,
            max_P=self.max_P,
            max_Q=self.max_Q,
            m=self.m,
            error_action='ignore',
            suppress_warnings=True,
            stepwise=True,
            information_criterion='aic'
        )
        
        # Get the order and seasonal_order
        self.order = auto_model.order
        self.seasonal_order = auto_model.seasonal_order if self.seasonal else None
        
        print(f"Best ARIMA parameters - Order: {self.order}, Seasonal Order: {self.seasonal_order}")
        
        # Fit the model with the best parameters
        if self.seasonal_order:
            self.model = SARIMAX(
                ts_data,
                order=self.order,
                seasonal_order=self.seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)
        else:
            self.model = ARIMA(
                ts_data,
                order=self.order
            ).fit()
        
        self.is_fitted = True
        return self

class EnsembleModel(BaseSignalModel):
    """
    Ensemble model that combines predictions from multiple models
    """
    def __init__(self, models=None, weights=None):
        super().__init__(name="Ensemble")
        self.models = models or []
        self.weights = weights
        
        # Normalize weights if provided
        if self.weights is not None:
            self.weights = np.array(self.weights) / sum(self.weights)
    
    def add_model(self, model, weight=1.0):
        """
        Add a model to the ensemble
        
        Args:
            model: Model instance
            weight: Weight for this model's predictions
        """
        self.models.append(model)
        
        # Update weights
        if self.weights is None:
            self.weights = np.ones(len(self.models))
        else:
            self.weights = np.append(self.weights, weight)
        
        # Normalize weights
        self.weights = self.weights / sum(self.weights)
    
    def fit(self, X, y):
        """
        Fit all models in the ensemble
        
        Args:
            X: Features array or DataFrame
            y: Target array or Series
        """
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        for i, model in enumerate(self.models):
            print(f"Training model {i+1}/{len(self.models)}: {model.name}")
            model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """
        Make weighted ensemble predictions
        
        Args:
            X: Features array or DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        # Get predictions from all models
        predictions = []
        for model in self.models:
            pred = model.predict(X)
            predictions.append(pred)
        
        # Convert predictions to probabilities
        probas = self.predict_proba(X)
        
        # Get the class with highest probability
        if len(probas.shape) == 2:  # Multi-class
            pred_class = np.argmax(probas, axis=1)
            
            # Map class index to signal value
            mapped_preds = np.zeros_like(pred_class)
            for i, cls in enumerate(pred_class):
                if cls == 0:
                    mapped_preds[i] = 0  # NEUTRAL
                elif cls == 1:
                    mapped_preds[i] = 1  # UP
                else:
                    mapped_preds[i] = -1  # DOWN
                
            return mapped_preds
        else:  # Binary class
            return np.where(probas > 0.5, 1, -1)
    
    def predict_proba(self, X):
        """
        Predict weighted class probabilities
        
        Args:
            X: Features array or DataFrame
            
        Returns:
            Array of class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        if len(self.models) == 0:
            raise ValueError("No models in ensemble")
        
        # Get probabilities from all models
        all_probas = []
        for i, model in enumerate(self.models):
            proba = model.predict_proba(X)
            
            # Handle different output shapes
            if len(proba.shape) == 1:  # Binary output with only positive class probability
                proba = np.column_stack((1 - proba, proba))
            
            # Weight the probabilities
            weighted_proba = proba * self.weights[i]
            all_probas.append(weighted_proba)
        
        # Sum the weighted probabilities
        ensemble_probas = sum(all_probas)
        
        # Normalize to ensure they sum to 1
        ensemble_probas = ensemble_probas / np.sum(ensemble_probas, axis=1, keepdims=True)
        
        return ensemble_probas
    
    def save(self, filepath):
        """
        Save the ensemble model and all its submodels
        
        Args:
            filepath: Path to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet")
        
        # Save each individual model
        model_paths = []
        for i, model in enumerate(self.models):
            model_path = f"{filepath}_model{i}_{model.name}.pkl"
            model.save(model_path)
            model_paths.append(model_path)
        
        # Save ensemble information
        ensemble_info = {
            'model_paths': model_paths,
            'weights': self.weights,
            'name': self.name
        }
        
        joblib.dump(ensemble_info, filepath)
        print(f"Ensemble model saved to {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """
        Load an ensemble model and all its submodels
        
        Args:
            filepath: Path to load the model from
            
        Returns:
            Loaded model
        """
        # Load ensemble information
        ensemble_info = joblib.load(filepath)
        
        # Create a new ensemble
        ensemble = cls(weights=ensemble_info['weights'])
        ensemble.name = ensemble_info['name']
        
        # Load each submodel
        for model_path in ensemble_info['model_paths']:
            # Determine the model type from the filename
            if 'ARIMA' in model_path:
                from .ts_models import ARIMAModel, AutoARIMAModel
                if 'Auto' in model_path:
                    model = AutoARIMAModel.load(model_path)
                else:
                    model = ARIMAModel.load(model_path)
            elif 'LSTM' in model_path or 'GRU' in model_path or 'CNN' in model_path:
                from .dl_models import LSTMModel, GRUModel, CNNLSTMModel
                if 'GRU' in model_path:
                    model = GRUModel.load(model_path)
                elif 'CNN' in model_path:
                    model = CNNLSTMModel.load(model_path)
                else:
                    model = LSTMModel.load(model_path)
            else:
                # Assume it's a standard ML model
                model = BaseSignalModel.load(model_path)
            
            ensemble.models.append(model)
        
        ensemble.is_fitted = True
        return ensemble 