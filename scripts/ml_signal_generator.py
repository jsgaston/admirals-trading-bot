# scripts/ml_signal_generator.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import talib as ta
from google_drive_manager import GoogleDriveManager

class MLSignalGenerator:
    def __init__(self):
        self.pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY']
        self.drive_manager = GoogleDriveManager()
        self.models = {}
        self.scalers = {}
        
    def load_historical_data(self, pair):
        """Cargar datos históricos desde Google Drive"""
        try:
            # Descargar datos desde Drive
            data = self.drive_manager.download_pair_data(pair)
            
            if not data:
                print(f"No data found for {pair}")
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            # Convertir a numérico
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col])
            
            return df
            
        except Exception as e:
            print(f"Error loading data for {pair}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df):
        """Calcular indicadores técnicos"""
        try:
            # Medias móviles
            df['sma_10'] = ta.SMA(df['close'], timeperiod=10)
            df['sma_20'] = ta.SMA(df['close'], timeperiod=20)
            df['sma_50'] = ta.SMA(df['close'], timeperiod=50)
            df['ema_12'] = ta.EMA(df['close'], timeperiod=12)
            df['ema_26'] = ta.EMA(df['close'], timeperiod=26)
            
            # Osciladores
            df['rsi'] = ta.RSI(df['close'], timeperiod=14)
            df['macd'], df['macd_signal'], df['macd_hist'] = ta.MACD(df['close'])
            df['stoch_k'], df['stoch_d'] = ta.STOCH(df['high'], df['low'], df['close'])
            
            # Bollinger Bands
            df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(df['close'])
            
            # Volatilidad
            df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Features de precio
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Features de volumen
            df['volume_sma'] = ta.SMA(df['volume'], timeperiod=20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Features avanzados
            df['williams_r'] = ta.WILLR(df['high'], df['low'], df['close'])
            df['cci'] = ta.CCI(df['high'], df['low'], df['close'])
            df['momentum'] = ta.MOM(df['close'], timeperiod=10)
            
            return df.dropna()
            
        except Exception as e:
            print(f"Error calculating technical indicators: {e}")
            return df
    
    def create_labels(self, df, forward_periods=4):
        """Crear labels para ML (precio en próximas 4 horas)"""
        try:
            # Calcular cambio de precio futuro
            df['future_price'] = df['close'].shift(-forward_periods)
            df['price_change_future'] = (df['future_price'] - df['close']) / df['close']
            
            # Crear labels: 0=HOLD, 1=BUY, 2=SELL
            df['label'] = 0  # HOLD por defecto
            
            # Thresholds más conservadores
            buy_threshold = 0.002  # 0.2%
            sell_threshold = -0.002  # -0.2%
            
            df.loc[df['price_change_future'] > buy_threshold, 'label'] = 1  # BUY
            df.loc[df['price_change_future'] < sell_threshold, 'label'] = 2  # SELL
            
            return df.dropna()
            
        except Exception as e:
            print(f"Error creating labels: {e}")
            return df
    
    def train_model(self, df, pair):
        """Entrenar modelo ML para un par específico"""
        try:
            # Features para el modelo
            feature_columns = [
                'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
                'bb_upper', 'bb_lower', 'atr', 'price_change',
                'high_low_ratio', 'close_position', 'volume_ratio',
                'williams_r', 'cci', 'momentum'
            ]
            
            # Preparar datos
            X = df[feature_columns].fillna(0)
            y = df['label']
            
            # Verificar que tenemos suficientes datos
            if len(X) < 100:
                print(f"Insufficient data for {pair}: {len(X)} samples")
                return None, None
            
            # Split train/test
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Normalizar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entrenar modelo
            model = RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluar
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            print(f"{pair} - Train: {train_score:.3f}, Test: {test_score:.3f}")
            
            # Guardar modelo y scaler
            self.models[pair] = model
            self.scalers[pair] = scaler
            
            return model, scaler
            
        except Exception as e:
            print(f"Error training model for {pair}: {e}")
            return None, None
    
    def generate_signal(self, pair, current_features):
        """Generar señal para un par"""
        try:
            if pair not in self.models or pair not in self.scalers:
                return {'signal': 'HOLD', 'confidence': 0}
            
            model = self.models[pair]
            scaler = self.scalers[pair]
            
            # Preparar features
            features = np.array(current_features).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            # Predicción
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
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
            print(f"Error generating signal for {pair}: {e}")
            return {'signal': 'HOLD', 'confidence': 0}
    
    def calculate_position_size(self, pair, signal, account_balance=10000):
        """Calcular tamaño de posición basado en riesgo"""
        try:
            risk_per_trade = 0.02  # 2% del capital
            
            # ATR para stop loss dinámico (simulado)
            atr_pips = 20 if 'JPY' in pair else 0.002
            
            risk_amount = account_balance * risk_per_trade
            
            if atr_pips == 0:
                return 0.01  # Tamaño mínimo
            
            # Calcular lots
            pip_value = 1 if 'JPY' in pair else 10  # USD per pip for 1 lot
            stop_loss_amount = atr_pips * pip_value
            
            position_size = risk_amount / stop_loss_amount
            
            # Límites
            min_lot = 0.01
            max_lot = 1.0
            
            return max(min_lot, min(max_lot, round(position_size, 2)))
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.01
    
    def generate_all_signals(self):
        """Generar señales para todos los pares"""
        signals = {
            'timestamp': datetime.now().isoformat(),
            'signals': {},
            'metadata': {
                'model_type': 'RandomForest',
                'features_count': 20,
                'confidence_threshold': 0.6
            }
        }
        
        for pair in self.pairs:
            try:
                print(f"Processing {pair}...")
                
                # Cargar y procesar datos
                df = self.load_historical_data(pair)
                
                if df.empty:
                    continue
                
                # Calcular indicadores técnicos
                df_with_indicators = self.calculate_technical_indicators(df)
                
                if len(df_with_indicators) < 100:
                    continue
                
                # Crear labels y entrenar modelo
                df_with_labels = self.create_labels(df_with_indicators)
                model, scaler = self.train_model(df_with_labels, pair)
                
                if model and scaler:
                    # Obtener features actuales (última fila)
                    feature_columns = [
                        'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
                        'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
                        'bb_upper', 'bb_lower', 'atr', 'price_change',
                        'high_low_ratio', 'close_position', 'volume_ratio',
                        'williams_r', 'cci', 'momentum'
                    ]
                    
                    current_features = df_with_indicators[feature_columns].iloc[-1].fillna(0).values
                    
                    # Generar señal
                    signal = self.generate_signal(pair, current_features)
                    
                    # Solo incluir señales con confianza > 60%
                    if signal['confidence'] > 0.6:
                        position_size = self.calculate_position_size(pair, signal)
                        
                        signals['signals'][pair] = {
                            'signal': signal['signal'],
                            'confidence': signal['confidence'],
                            'probabilities': signal['probabilities'],
                            'position_size': position_size,
                            'current_price': float(df_with_indicators['close'].iloc[-1]),
                            'rsi': float(df_with_indicators['rsi'].iloc[-1]),
                            'macd': float(df_with_indicators['macd'].iloc[-1])
                        }
                
            except Exception as e:
                print(f"Error processing {pair}: {e}")
        
        # Guardar señales
        self.save_signals(signals)
        
        return signals
    
    def save_signals(self, signals):
        """Guardar señales generadas"""
        try:
            os.makedirs('data', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Guardar localmente
            with open(f'data/signals_{timestamp}.json', 'w') as f:
                json.dump(signals, f, indent=2)
            
            with open('data/latest_signals.json', 'w') as f:
                json.dump(signals, f, indent=2)
            
            # Subir a Google Drive
            self.drive_manager.upload_signals(signals)
            
            print(f"Signals saved: {len(signals['signals'])} pairs")
            
        except Exception as e:
            print(f"Error saving signals: {e}")

if __name__ == "__main__":
    generator = MLSignalGenerator()
    signals = generator.generate_all_signals()
    
    print(f"Generated signals for {len(signals['signals'])} pairs")
    for pair, signal_data in signals['signals'].items():
        print(f"{pair}: {signal_data['signal']} ({signal_data['confidence']:.3f})")
    
    exit(0 if len(signals['signals']) > 0 else 1)

---
