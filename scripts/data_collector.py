# scripts/data_collector.py
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from google_drive_manager import GoogleDriveManager

class ForexDataCollector:
    def __init__(self):
        self.session = requests.Session()
        self.pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY']
        self.api_base = "https://api.exchangerate-api.com/v4"
        self.drive_manager = GoogleDriveManager()
        
    def get_current_rates(self):
        """Obtener tasas actuales de forex"""
        try:
            url = f"{self.api_base}/latest/EUR"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            rates = data.get('rates', {})
            print(f"Retrieved rates for {len(rates)} currencies")
            return rates
            
        except Exception as e:
            print(f"Error fetching rates: {e}")
            return {}
    
    def calculate_pair_rate(self, pair, base_rates):
        """Calcular tasa para un par específico"""
        try:
            base_currency = pair[:3]
            quote_currency = pair[3:]
            
            if base_currency == 'EUR':
                return base_rates.get(quote_currency)
            elif quote_currency == 'EUR':
                base_rate = base_rates.get(base_currency)
                return 1 / base_rate if base_rate else None
            else:
                base_rate = base_rates.get(base_currency)
                quote_rate = base_rates.get(quote_currency)
                return base_rate / quote_rate if base_rate and quote_rate else None
                
        except Exception as e:
            print(f"Error calculating rate for {pair}: {e}")
            return None
    
    def generate_historical_ohlc(self, current_rate, hours=168, pair='EURUSD'):
        """Generar datos OHLC históricos realistas"""
        ohlc_data = []
        price = current_rate
        
        for i in range(hours):
            # Volatilidad realista por hora
            base_volatility = 0.003 if 'JPY' in pair else 0.002
            
            # Componentes de movimiento
            random_walk = np.random.normal(0, base_volatility)
            mean_reversion = (current_rate - price) * 0.005
            trend_component = np.random.normal(0, base_volatility / 4)
            
            # Ajuste por sesión
            hour_of_day = (datetime.now() - timedelta(hours=hours-i-1)).hour
            if 8 <= hour_of_day <= 17:  # Sesión europea/americana
                volatility_multiplier = 1.5
            else:  # Sesión asiática
                volatility_multiplier = 0.7
            
            # Calcular nuevo precio
            total_change = (random_walk + mean_reversion + trend_component) * volatility_multiplier
            new_price = price * (1 + total_change)
            
            # Generar OHLC
            intrabar_volatility = new_price * base_volatility * volatility_multiplier
            high_offset = np.random.uniform(0, intrabar_volatility)
            low_offset = np.random.uniform(0, intrabar_volatility)
            
            high = new_price + high_offset
            low = new_price - low_offset
            
            open_price = price
            close_price = new_price
            
            final_high = max(open_price, close_price, high)
            final_low = min(open_price, close_price, low)
            
            # Spread realista
            pip_value = 0.01 if 'JPY' in pair else 0.0001
            spread = np.random.uniform(1.0, 3.0) * pip_value
            
            ohlc_data.append({
                'timestamp': (datetime.now() - timedelta(hours=hours-i-1)).isoformat(),
                'open': round(open_price, 5),
                'high': round(final_high, 5),
                'low': round(final_low, 5),
                'close': round(close_price, 5),
                'volume': np.random.randint(1000, 10000),
                'spread': round(spread, 5),
                'pair': pair
            })
            
            price = new_price
        
        return ohlc_data
    
    def collect_all_data(self):
        """Recolectar datos de todos los pares"""
        print(f"Starting data collection at {datetime.now()}")
        
        try:
            current_rates = self.get_current_rates()
            if not current_rates:
                print("No current rates available")
                return False
            
            all_data = {}
            
            for pair in self.pairs:
                try:
                    current_rate = self.calculate_pair_rate(pair, current_rates)
                    
                    if current_rate:
                        # Generar 1 semana de datos históricos
                        ohlc_data = self.generate_historical_ohlc(current_rate, hours=168, pair=pair)
                        all_data[pair] = ohlc_data
                        print(f"Generated {len(ohlc_data)} data points for {pair}")
                        
                except Exception as e:
                    print(f"Error processing {pair}: {e}")
            
            # Guardar localmente
            self.save_data_locally(all_data)
            
            # Subir a Google Drive
            self.drive_manager.upload_forex_data(all_data)
            
            print(f"Collection completed: {len(all_data)} pairs processed")
            return len(all_data) > 0
            
        except Exception as e:
            print(f"Error in data collection: {e}")
            return False
    
    def save_data_locally(self, data):
        """Guardar datos localmente"""
        try:
            os.makedirs('data', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for pair, pair_data in data.items():
                filename = f"data/{pair}_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump({
                        'pair': pair,
                        'timestamp': datetime.now().isoformat(),
                        'data': pair_data
                    }, f, indent=2)
            
            # Resumen
            summary = {
                'collection_timestamp': datetime.now().isoformat(),
                'pairs_collected': list(data.keys()),
                'total_data_points': sum(len(pair_data) for pair_data in data.values())
            }
            
            with open('data/collection_summary.json', 'w') as f:
                json.dump(summary, f, indent=2)
            
            print("Data saved locally successfully")
            
        except Exception as e:
            print(f"Error saving data locally: {e}")

if __name__ == "__main__":
    collector = ForexDataCollector()
    success = collector.collect_all_data()
    exit(0 if success else 1)

---

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

# scripts/myfxbook_sender.py
import requests
import json
import os
from datetime import datetime

class MyFxBookSender:
    def __init__(self):
        self.api_key = os.environ.get('MYFXBOOK_API_KEY')
        self.webhook_url = os.environ.get('MYFXBOOK_WEBHOOK_URL')
        self.account_id = os.environ.get('MYFXBOOK_ACCOUNT_ID')
        self.session = requests.Session()
        
    def login_to_myfxbook(self):
        """Autenticar con MyFxBook API"""
        try:
            login_url = "https://www.myfxbook.com/api/login.json"
            
            payload = {
                'email': os.environ.get('MYFXBOOK_EMAIL'),
                'password': os.environ.get('MYFXBOOK_PASSWORD')
            }
            
            response = self.session.post(login_url, data=payload)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('error') == False:
                    self.session_id = data.get('session')
                    print("MyFxBook login successful")
                    return True
            
            print("MyFxBook login failed")
            return False
            
        except Exception as e:
            print(f"Error logging into MyFxBook: {e}")
            return False
    
    def send_signal_to_autotrade(self, pair, signal_data):
        """Enviar señal individual a MyFxBook AutoTrade"""
        try:
            if not self.webhook_url:
                print("MyFxBook webhook URL not configured")
                return False
            
            # Preparar payload para MyFxBook
            payload = {
                'symbol': pair,
                'action': signal_data['signal'].lower(),
                'volume': signal_data['position_size'],
                'comment': f"ML_Signal_Conf_{signal_data['confidence']:.2f}",
                'timestamp': datetime.now().isoformat()
            }
            
            # Calcular stop loss y take profit dinámicos
            current_price = signal_data['current_price']
            
            if signal_data['signal'] == 'BUY':
                # Para BUY: SL abajo, TP arriba
                atr_distance = current_price * 0.002  # 0.2% como proxy de ATR
                payload['sl'] = round(current_price - (atr_distance * 1.5), 5)
                payload['tp'] = round(current_price + (atr_distance * 2.0), 5)
                
            elif signal_data['signal'] == 'SELL':
                # Para SELL: SL arriba, TP abajo
                atr_distance = current_price * 0.002
                payload['sl'] = round(current_price + (atr_distance * 1.5), 5)
                payload['tp'] = round(current_price - (atr_distance * 2.0), 5)
            
            # Enviar a MyFxBook webhook
            response = requests.post(self.webhook_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                print(f"Signal sent to MyFxBook: {pair} {signal_data['signal']}")
                return True
            else:
                print(f"MyFxBook webhook failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            print(f"Error sending signal to MyFxBook: {e}")
            return False
    
    def send_all_signals(self):
        """Enviar todas las señales generadas a MyFxBook"""
        try:
            # Cargar señales generadas
            with open('data/latest_signals.json', 'r') as f:
                signals_data = json.load(f)
            
            signals = signals_data.get('signals', {})
            
            if not signals:
                print("No signals to send")
                return False
            
            # Autenticar con MyFxBook (si es necesario)
            if self.api_key:
                self.login_to_myfxbook()
            
            sent_count = 0
            
            for pair, signal_data in signals.items():
                if signal_data['signal'] in ['BUY', 'SELL']:
                    if self.send_signal_to_autotrade(pair, signal_data):
                        sent_count += 1
            
            print(f"Sent {sent_count}/{len(signals)} signals to MyFxBook")
            
            # Guardar log de envío
            self.save_sending_log(signals, sent_count)
            
            return sent_count > 0
            
        except Exception as e:
            print(f"Error in send_all_signals: {e}")
            return False
    
    def save_sending_log(self, signals, sent_count):
        """Guardar log de envío"""
        try:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'total_signals': len(signals),
                'signals_sent': sent_count,
                'signals_detail': {
                    pair: {
                        'signal': data['signal'],
                        'confidence': data['confidence'],
                        'position_size': data['position_size']
                    }
                    for pair, data in signals.items()
                }
            }
            
            # Guardar log
            os.makedirs('logs', exist_ok=True)
            log_file = f"logs/myfxbook_sending_{datetime.now().strftime('%Y%m%d')}.json"
            
            # Cargar logs existentes
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    existing_logs = json.load(f)
            else:
                existing_logs = []
            
            existing_logs.append(log_entry)
            
            with open(log_file, 'w') as f:
                json.dump(existing_logs, f, indent=2)
            
            print(f"Sending log saved to {log_file}")
            
        except Exception as e:
            print(f"Error saving sending log: {e}")
    
    def check_myfxbook_status(self):
        """Verificar estado de la cuenta en MyFxBook"""
        try:
            if not self.session_id:
                return None
            
            status_url = "https://www.myfxbook.com/api/get-my-accounts.json"
            params = {'session': self.session_id}
            
            response = self.session.get(status_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                return data.get('accounts', [])
            
            return None
            
        except Exception as e:
            print(f"Error checking MyFxBook status: {e}")
            return None

if __name__ == "__main__":
    sender = MyFxBookSender()
    success = sender.send_all_signals()
    exit(0 if success else 1)

---

# scripts/google_drive_manager.py
import os
import json
import io
from datetime import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

class GoogleDriveManager:
    def __init__(self):
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER')
        self.service = self._authenticate()
        
    def _authenticate(self):
        """Autenticar con Google Drive"""
        try:
            credentials_json = os.environ.get('GOOGLE_CREDENTIALS')
            
            if not credentials_json:
                print("Google credentials not found")
                return None
            
            credentials_info = json.loads(credentials_json)
            credentials = Credentials.from_service_account_info(
                credentials_info,
                scopes=self.scopes
            )
            
            return build('drive', 'v3', credentials=credentials)
            
        except Exception as e:
            print(f"Error authenticating Google Drive: {e}")
            return None
    
    def upload_forex_data(self, data):
        """Subir datos forex a Google Drive"""
        try:
            if not self.service:
                return False
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for pair, pair_data in data.items():
                # Preparar archivo JSON
                file_content = {
                    'pair': pair,
                    'timestamp': datetime.now().isoformat(),
                    'data_points': len(pair_data),
                    'data': pair_data
                }
                
                # Convertir a bytes
                file_buffer = io.BytesIO()
                file_buffer.write(json.dumps(file_content, indent=2).encode())
                file_buffer.seek(0)
                
                # Metadata del archivo
                file_metadata = {
                    'name': f'{pair}_data_{timestamp}.json',
                    'parents': [self.folder_id] if self.folder_id else []
                }
                
                # Upload
                media = MediaIoBaseUpload(
                    file_buffer,
                    mimetype='application/json',
                    resumable=True
                )
                
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                
                print(f"Uploaded {pair} data to Drive: {file.get('id')}")
            
            return True
            
        except Exception as e:
            print(f"Error uploading forex data: {e}")
            return False
    
    def upload_signals(self, signals):
        """Subir señales a Google Drive"""
        try:
            if not self.service:
                return False
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Preparar archivo
            file_buffer = io.BytesIO()
            file_buffer.write(json.dumps(signals, indent=2).encode())
            file_buffer.seek(0)
            
            # Metadata
            file_metadata = {
                'name': f'trading_signals_{timestamp}.json',
                'parents': [self.folder_id] if self.folder_id else []
            }
            
            # Upload
            media = MediaIoBaseUpload(
                file_buffer,
                mimetype='application/json',
                resumable=True
            )
            
            # Verificar si existe archivo "latest_signals.json"
            existing_file = self._find_file('latest_signals.json')
            
            if existing_file:
                # Actualizar archivo existente
                updated_file = self.service.files().update(
                    fileId=existing_file['id'],
                    media_body=media
                ).execute()
                print(f"Updated latest signals in Drive")
            else:
                # Crear nuevo archivo
                file = self.service.files().create(
                    body={**file_metadata, 'name': 'latest_signals.json'},
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"Created new signals file in Drive")
            
            # También crear archivo con timestamp
            file_with_timestamp = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            return True
            
        except Exception as e:
            print(f"Error uploading signals: {e}")
            return False
    
    def download_pair_data(self, pair):
        """Descargar datos de un par específico"""
        try:
            if not self.service:
                return []
            
            # Buscar archivos del par
            query = f"name contains '{pair}_data_' and parents in '{self.folder_id}'"
            
            results = self.service.files().list(
                q=query,
                orderBy='modifiedTime desc',
                pageSize=1,
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            
            if not files:
                print(f"No data files found for {pair}")
                return []
            
            # Descargar el archivo más reciente
            file_id = files[0]['id']
            
            request = self.service.files().get_media(fileId=file_id)
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # Parsear JSON
            file_buffer.seek(0)
            file_content = json.loads(file_buffer.read().decode())
            
            return file_content.get('data', [])
            
        except Exception as e:
            print(f"Error downloading data for {pair}: {e}")
            return []
    
    def _find_file(self, filename):
        """Buscar archivo por nombre"""
        try:
            query = f"name='{filename}'"
            if self.folder_id:
                query += f" and parents in '{self.folder_id}'"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            return files[0] if files else None
            
        except Exception as e:
            print(f"Error finding file {filename}: {e}")
            return None
    
    def cleanup_old_files(self, days_old=7):
        """Limpiar archivos antiguos"""
        try:
            if not self.service:
                return False
            
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat()
            
            query = f"modifiedTime < '{cutoff_str}'"
            if self.folder_id:
                query += f" and parents in '{self.folder_id}'"
            
            results = self.service.files().list(q=query).execute()
            files = results.get('files', [])
            
            deleted_count = 0
            for file in files:
                self.service.files().delete(fileId=file['id']).execute()
                deleted_count += 1
            
            print(f"Cleaned up {deleted_count} old files")
            return True
            
        except Exception as e:
            print(f"Error cleaning up files: {e}")
            return False

if __name__ == "__main__":
    manager = GoogleDriveManager()
    
    # Test de funcionalidad
    if manager.service:
        print("Google Drive connection successful")
    else:
        print("Google Drive connection failed")
