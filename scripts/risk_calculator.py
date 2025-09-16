# scripts/risk_calculator.py
# Cálculo de stop loss y take profit dinámicos

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import talib as ta

class RiskCalculator:
    """Calculador de riesgo con múltiples timeframes"""
    
    def __init__(self):
        self.session = requests.Session()
        self.risk_per_trade = 0.02  # 2% del capital por trade
        self.max_risk_per_day = 0.06  # 6% del capital por día
        
    def get_current_data_15m(self, pair):
        """Obtener datos de 15M para cálculos de riesgo"""
        try:
            # En implementación real, obtener datos reales de 15M
            # Por ahora, simular datos basados en datos hourly
            
            # Simular 96 períodos de 15M (24 horas)
            data_15m = []
            base_price = self.get_current_price(pair)
            
            if not base_price:
                return []
            
            price = base_price
            
            for i in range(96):  # 96 períodos de 15M = 24 horas
                # Volatilidad de 15M (menor que 1H)
                volatility = 0.0005  # 0.05% por período de 15M
                
                # Movimiento aleatorio
                change = np.random.normal(0, volatility)
                price *= (1 + change)
                
                # Generar OHLC
                high = price * (1 + abs(np.random.normal(0, volatility/2)))
                low = price * (1 - abs(np.random.normal(0, volatility/2)))
                
                data_15m.append({
                    'timestamp': (datetime.now() - timedelta(minutes=15*(96-i-1))).isoformat(),
                    'open': price,
                    'high': max(price, high),
                    'low': min(price, low),
                    'close': price,
                    'volume': np.random.randint(500, 2000)
                })
            
            return data_15m
            
        except Exception as e:
            print(f"Error getting 15M data for {pair}: {e}")
            return []
    
    def get_current_price(self, pair):
        """Obtener precio actual del par"""
        try:
            # Usar ExchangeRate API
            url = "https://api.exchangerate-api.com/v4/latest/EUR"
            response = self.session.get(url, timeout=30)
            data = response.json()
            rates = data.get('rates', {})
            
            # Calcular tasa del par
            base_currency = pair[:3]
            quote_currency = pair[3:]
            
            if base_currency == 'EUR':
                return rates.get(quote_currency)
            elif quote_currency == 'EUR':
                base_rate = rates.get(base_currency)
                return 1 / base_rate if base_rate else None
            else:
                base_rate = rates.get(base_currency)
                quote_rate = rates.get(quote_currency)
                return base_rate / quote_rate if base_rate and quote_rate else None
                
        except Exception as e:
            print(f"Error getting current price for {pair}: {e}")
            return None
    
    def calculate_atr_stop_loss(self, data_15m, atr_multiplier=1.5):
        """Calcular stop loss basado en ATR de 15M"""
        try:
            if len(data_15m) < 14:
                return None
            
            # Convertir a DataFrame
            df = pd.DataFrame(data_15m)
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['close'] = pd.to_numeric(df['close'])
            
            # Calcular ATR
            atr = ta.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
            current_atr = atr[-1]
            current_price = df['close'].iloc[-1]
            
            # Stop loss basado en ATR
            stop_distance = current_atr * atr_multiplier
            
            return {
                'atr_value': current_atr,
                'stop_distance': stop_distance,
                'stop_loss_long': current_price - stop_distance,
                'stop_loss_short': current_price + stop_distance
            }
            
        except Exception as e:
            print(f"Error calculating ATR stop loss: {e}")
            return None
    
    def calculate_volatility_stop(self, data_15m, lookback=20):
        """Calcular stop loss basado en volatilidad histórica"""
        try:
            if len(data_15m) < lookback:
                return None
            
            df = pd.DataFrame(data_15m)
            df['close'] = pd.to_numeric(df['close'])
            
            # Calcular returns
            df['returns'] = df['close'].pct_change()
            
            # Volatilidad histórica
            volatility = df['returns'].rolling(window=lookback).std().iloc[-1]
            current_price = df['close'].iloc[-1]
            
            # Stop loss a 2 desviaciones estándar
            stop_distance = current_price * volatility * 2
            
            return {
                'volatility': volatility,
                'stop_distance': stop_distance,
                'stop_loss_long': current_price - stop_distance,
                'stop_loss_short': current_price + stop_distance
            }
            
        except Exception as e:
            print(f"Error calculating volatility stop: {e}")
            return None
    
    def calculate_dynamic_take_profit(self, entry_price, stop_loss, risk_reward_ratio=2.0):
        """Calcular take profit dinámico"""
        try:
            stop_distance = abs(entry_price - stop_loss)
            profit_distance = stop_distance * risk_reward_ratio
            
            if entry_price > stop_loss:  # Long position
                take_profit = entry_price + profit_distance
            else:  # Short position
                take_profit = entry_price - profit_distance
            
            return {
                'take_profit': take_profit,
                'risk_reward_ratio': risk_reward_ratio,
                'profit_distance': profit_distance
            }
            
        except Exception as e:
            print(f"Error calculating take profit: {e}")
            return None
    
    def calculate_position_size(self, account_balance, entry_price, stop_loss):
        """Calcular tamaño de posición basado en riesgo"""
        try:
            risk_amount = account_balance * self.risk_per_trade
            stop_distance = abs(entry_price - stop_loss)
            
            if stop_distance == 0:
                return None
            
            # Calcular tamaño de posición
            position_size = risk_amount / stop_distance
            
            # Para forex, convertir a lots
            standard_lot = 100000
            position_lots = position_size / standard_lot
            
            return {
                'position_size': position_size,
                'position_lots': round(position_lots, 2),
                'risk_amount': risk_amount,
                'stop_distance': stop_distance
            }
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return None
    
    def assess_all_pairs_risk(self, pairs, account_balance=10000):
        """Evaluar riesgo de todos los pares"""
        
        risk_assessment = {
            'timestamp': datetime.now().isoformat(),
            'account_balance': account_balance,
            'max_daily_risk': account_balance * self.max_risk_per_day,
            'pairs': {}
        }
        
        for pair in pairs:
            try:
                print(f"Calculating risk for {pair}...")
                
                # Obtener datos de 15M
                data_15m = self.get_current_data_15m(pair)
                
                if not data_15m:
                    continue
                
                current_price = float(data_15m[-1]['close'])
                
                # Calcular diferentes tipos de stop loss
                atr_stop = self.calculate_atr_stop_loss(data_15m)
                vol_stop = self.calculate_volatility_stop(data_15m)
                
                if atr_stop and vol_stop:
                    # Usar el stop loss más conservador
                    stop_loss_long = max(atr_stop['stop_loss_long'], vol_stop['stop_loss_long'])
                    stop_loss_short = min(atr_stop['stop_loss_short'], vol_stop['stop_loss_short'])
                    
                    # Calcular take profit
                    tp_long = self.calculate_dynamic_take_profit(current_price, stop_loss_long)
                    tp_short = self.calculate_dynamic_take_profit(current_price, stop_loss_short)
                    
                    # Calcular tamaño de posición
                    position_size_long = self.calculate_position_size(account_balance, current_price, stop_loss_long)
                    position_size_short = self.calculate_position_size(account_balance, current_price, stop_loss_short)
                    
                    risk_assessment['pairs'][pair] = {
                        'current_price': current_price,
                        'long_setup': {
                            'stop_loss': stop_loss_long,
                            'take_profit': tp_long['take_profit'] if tp_long else None,
                            'position_size': position_size_long['position_lots'] if position_size_long else None,
                            'risk_amount': position_size_long['risk_amount'] if position_size_long else None
                        },
                        'short_setup': {
                            'stop_loss': stop_loss_short,
                            'take_profit': tp_short['take_profit'] if tp_short else None,
                            'position_size': position_size_short['position_lots'] if position_size_short else None,
                            'risk_amount': position_size_short['risk_amount'] if position_size_short else None
                        },
                        'atr_data': atr_stop,
                        'volatility_data': vol_stop
                    }
                
            except Exception as e:
                print(f"Error assessing risk for {pair}: {e}")
        
        # Guardar assessment
        self.save_risk_assessment(risk_assessment)
        
        return risk_assessment
    
    def save_risk_assessment(self, assessment):
        """Guardar evaluación de riesgo"""
        try:
            import os
            os.makedirs('data', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/risk_assessment_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(assessment, f, indent=2)
            
            # También guardar como latest
            with open('data/latest_risk_assessment.json', 'w') as f:
                json.dump(assessment, f, indent=2)
            
            print(f"Risk assessment saved to {filename}")
            
        except Exception as e:
            print(f"Error saving risk assessment: {e}")

def main():
    """Función principal para GitHub Actions"""
    
    try:
        calculator = RiskCalculator()
        pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        
        # Calcular riesgo para todos los pares
        assessment = calculator.assess_all_pairs_risk(pairs)
        
        print(f"Risk assessment completed for {len(assessment['pairs'])} pairs")
        
        # Mostrar resumen
        for pair, data in assessment['pairs'].items():
            print(f"\n{pair}:")
            print(f"  Current Price: {data['current_price']}")
            print(f"  Long SL: {data['long_setup']['stop_loss']:.5f}")
            print(f"  Short SL: {data['short_setup']['stop_loss']:.5f}")
        
        return True
        
    except Exception as e:
        print(f"Error in risk calculation: {e}")
        return False

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)

---

# scripts/entry_signal_generator.py
# Generador de señales de entrada usando datos de 1H

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class EntrySignalGenerator:
    """Generador de señales de entrada con ML simple"""
    
    def __init__(self):
        self.pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']
        self.lookback_hours = 720  # 30 días de datos horarios
        
    def load_historical_data(self, pair):
        """Cargar datos históricos del par"""
        try:
            # En implementación real, cargar desde Google Drive
            # Por ahora, simular datos
            
            data = []
            base_price = 1.1800 if pair == 'EURUSD' else 1.0000
            price = base_price
            
            for i in range(self.lookback_hours):
                # Simular datos históricos
                change = np.random.normal(0, 0.002)
                price *= (1 + change)
                
                high = price * (1 + abs(np.random.normal(0, 0.001)))
                low = price * (1 - abs(np.random.normal(0, 0.001)))
                
                data.append({
                    'timestamp': (datetime.now() - timedelta(hours=self.lookback_hours-i-1)).isoformat(),
                    'open': price,
                    'high': high,
                    'low': low,
                    'close': price,
                    'volume': np.random.randint(1000, 5000)
                })
            
            return data
            
        except Exception as e:
            print(f"Error loading historical data for {pair}: {e}")
            return []
    
    def calculate_features(self, data):
        """Calcular features técnicos"""
        try:
            df = pd.DataFrame(data)
            df['close'] = pd.to_numeric(df['close'])
            df['high'] = pd.to_numeric(df['high'])
            df['low'] = pd.to_numeric(df['low'])
            df['volume'] = pd.to_numeric(df['volume'])
            
            # Features técnicos básicos
            df['sma_20'] = df['close'].rolling(window=20).mean()
            df['sma_50'] = df['close'].rolling(window=50).mean()
            df['rsi'] = self.calculate_rsi(df['close'], 14)
            df['macd'] = df['close'].ewm(span=12).mean() - df['close'].ewm(span=26).mean()
            df['bb_upper'], df['bb_lower'] = self.calculate_bollinger_bands(df['close'])
            
            # Features de precio
            df['price_change'] = df['close'].pct_change()
            df['high_low_ratio'] = df['high'] / df['low']
            df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Features de volumen
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            return df.dropna()
            
        except Exception as e:
            print(f"Error calculating features: {e}")
            return pd.DataFrame()
    
    def calculate_rsi(self, prices, period=14):
        """Calcular RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_bollinger_bands(self, prices, period=20, std_dev=2):
        """Calcular Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower
    
    def create_labels(self, df, forward_periods=4):
        """Crear labels para ML (precio en 4 horas)"""
        try:
            # Label: 1 si el precio sube más del 0.1% en las próximas 4 horas, 0 si no
            df['future_price'] = df['close'].shift(-forward_periods)
            df['price_change_future'] = (df['future_price'] - df['close']) / df['close']
            
            # Labels: 0=Hold, 1=Buy, 2=Sell
            df['label'] = 0  # Hold por defecto
            df.loc[df['price_change_future'] > 0.001, 'label'] = 1  # Buy
            df.loc[df['price_change_future'] < -0.001, 'label'] = 2  # Sell
            
            return df.dropna()
            
        except Exception as e:
            print(f"Error creating labels: {e}")
            return df
    
    def train_model(self, df):
        """Entrenar modelo ML simple"""
        try:
            # Features para el modelo
            feature_columns = ['sma_20', 'sma_50', 'rsi', 'macd', 'bb_upper', 'bb_lower',
                             'price_change', 'high_low_ratio', 'close_position', 'volume_ratio']
            
            X = df[feature_columns].fillna(0)
            y = df['label']
            
            # Separar train/test
            split_point = int(len(X) * 0.8)
            X_train, X_test = X[:split_point], X[split_point:]
            y_train, y_test = y[:split_point], y[split_point:]
            
            # Normalizar features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Entrenar modelo
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, y_train)
            
            # Evaluar
            train_score = model.score(X_train_scaled, y_train)
            test_score = model.score(X_test_scaled, y_test)
            
            print(f"Train accuracy: {train_score:.3f}, Test accuracy: {test_score:.3f}")
            
            return model, scaler
            
        except Exception as e:
            print(f"Error training model: {e}")
            return None, None
    
    def generate_signal(self, model, scaler, current_features):
        """Generar señal para el momento actual"""
        try:
            # Preparar features
            features = np.array(current_features).reshape(1, -1)
            features_scaled = scaler.transform(features)
            
            # Predicción
            prediction = model.predict(features_scaled)[0]
            probability = model.predict_proba(features_scaled)[0]
            
            signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            
            return {
                'signal': signal_map[prediction],
                'confidence': max(probability),
                'probabilities': {
                    'hold': probability[0],
                    'buy': probability[1] if len(probability) > 1 else 0,
                    'sell': probability[2] if len(probability) > 2 else 0
                }
            }
            
        except Exception as e:
            print(f"Error generating signal: {e}")
            return {'signal': 'HOLD', 'confidence': 0}
    
    def generate_all_signals(self):
        """Generar señales para todos los pares"""
        
        signals = {
            'timestamp': datetime.now().isoformat(),
            'signals': {}
        }
        
        for pair in self.pairs:
            try:
                print(f"Generating signal for {pair}...")
                
                # Cargar datos históricos
                data = self.load_historical_data(pair)
                
                if not data:
                    continue
                
                # Calcular features
                df = self.calculate_features(data)
                
                if df.empty:
                    continue
                
                # Crear labels y entrenar modelo
                df_with_labels = self.create_labels(df)
                model, scaler = self.train_model(df_with_labels)
                
                if model and scaler:
                    # Obtener features actuales (última fila)
                    feature_columns = ['sma_20', 'sma_50', 'rsi', 'macd', 'bb_upper', 'bb_lower',
                                     'price_change', 'high_low_ratio', 'close_position', 'volume_ratio']
                    
                    current_features = df[feature_columns].iloc[-1].fillna(0).values
                    
                    # Generar señal
                    signal = self.generate_signal(model, scaler, current_features)
                    
                    signals['signals'][pair] = {
                        'current_price': float(df['close'].iloc[-1]),
                        'signal': signal['signal'],
                        'confidence': signal['confidence'],
                        'probabilities': signal['probabilities'],
                        'features': {col: float(df[col].iloc[-1]) for col in feature_columns}
                    }
                
            except Exception as e:
                print(f"Error generating signal for {pair}: {e}")
        
        # Guardar señales
        self.save_signals(signals)
        
        return signals
    
    def save_signals(self, signals):
        """Guardar señales generadas"""
        try:
            os.makedirs('data', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data/entry_signals_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(signals, f, indent=2)
            
            # También guardar como latest
            with open('data/latest_entry_signals.json', 'w') as f:
                json.dump(signals, f, indent=2)
            
            print(f"Signals saved to {filename}")
            
        except Exception as e:
            print(f"Error saving signals: {e}")

def main():
    """Función principal para GitHub Actions"""
    
    try:
        generator = EntrySignalGenerator()
        signals = generator.generate_all_signals()
        
        print(f"Generated signals for {len(signals['signals'])} pairs")
        
        # Mostrar resumen
        for pair, signal_data in signals['signals'].items():
            print(f"{pair}: {signal_data['signal']} (confidence: {signal_data['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"Error generating signals: {e}")
        return False

if __name__ == "__main__":
    success = main()
    import sys
    sys.exit(0 if success else 1)
