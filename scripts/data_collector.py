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
