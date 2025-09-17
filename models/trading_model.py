# models/trading_model.py
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

class TradingModel:
    """Modelo de trading con ML"""
    
    def __init__(self, pair):
        self.pair = pair
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
            'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
            'bb_upper', 'bb_lower', 'atr', 'price_change',
            'high_low_ratio', 'close_position', 'volume_ratio',
            'williams_r', 'cci', 'momentum'
        ]
    
    def prepare_features(self, df):
        """Preparar features para el modelo"""
        try:
            # Verificar que tenemos todas las columnas necesarias
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")
            
            # Seleccionar y limpiar features
            X = df[self.feature_columns].fillna(0)
            
            return X
            
        except Exception as e:
            print(f"Error preparing features: {e}")
            return None
    
    def create_labels(self, df, forward_periods=4, threshold=0.002):
        """Crear labels para entrenamiento"""
        try:
            # Calcular cambio de precio futuro
            df['future_price'] = df['close'].shift(-forward_periods)
            df['price_change_future'] = (df['future_price'] - df['close']) / df['close']
            
            # Crear labels: 0=HOLD, 1=BUY, 2=SELL
            labels = np.zeros(len(df))
            labels[df['price_change_future'] > threshold] = 1  # BUY
            labels[df['price_change_future'] < -threshold] = 2  # SELL
            
            return labels
            
        except Exception as e:
            print(f"Error creating labels: {e}")
            return None
    
    def train(self, df):
        """Entrenar el modelo"""
        try:
            # Preparar features
            X = self.prepare_features(df)
            if X is None:
                return False
            
            # Crear labels
            y = self.create_labels(df)
            if y is None:
                return False
            
            # Eliminar filas con NaN
            mask = ~np.isnan(y)
            X = X[mask]
            y = y[mask]
            
            if len(X) < 100:
                print(f"Insufficient data for training: {len(X)} samples")
                return False
            
            # Split train/test
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Normalizar features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Entrenar modelo
            self.model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluar
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            print(f"{self.pair} - Train: {train_score:.3f}, Test: {test_score:.3f}")
            
            return True
            
        except Exception as e:
            print(f"Error training model for {self.pair}: {e}")
            return False
    
    def predict(self, features):
        """Hacer predicción"""
        try:
            if self.model is None or self.scaler is None:
                return {'signal': 'HOLD', 'confidence': 0}
            
            # Preparar features
            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)
            
            # Predicción
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            
            return {
                'signal': signal_map[prediction],
                'confidence': max(probabilities),
                'probabilities': {
                    'hold': probabilities[0],
                    'buy': probabilities[1] if len(probabilities) > 1 else 0,
                    'sell': probabilities[2] if len(probabilities) > 2 else 0
                }
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            return {'signal': 'HOLD', 'confidence': 0}
    
    def save_model(self, path):
        """Guardar modelo entrenado"""
        try:
            os.makedirs(path, exist_ok=True)
            
            if self.model is not None:
                model_file = os.path.join(path, f'{self.pair}_model.joblib')
                joblib.dump(self.model, model_file)
            
            if self.scaler is not None:
                scaler_file = os.path.join(path, f'{self.pair}_scaler.joblib')
                joblib.dump(self.scaler, scaler_file)
            
            print(f"Model saved for {self.pair}")
            return True
            
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def load_model(self, path):
        """Cargar modelo entrenado"""
        try:
            model_file = os.path.join(path, f'{self.pair}_model.joblib')
            scaler_file = os.path.join(path, f'{self.pair}_scaler.joblib')
            
            if os.path.exists(model_file):
                self.model = joblib.load(model_file)
            
            if os.path.exists(scaler_file):
                self.scaler = joblib.load(scaler_file)
            
            print(f"Model loaded for {self.pair}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
