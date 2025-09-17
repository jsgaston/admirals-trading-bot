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
