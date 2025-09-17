# scripts/risk_manager.py
import json
import os
from datetime import datetime, timedelta
import requests

class RiskManager:
    def __init__(self):
        self.max_daily_risk = float(os.environ.get('MAX_DAILY_RISK', '5.0'))  # 5%
        self.max_drawdown = float(os.environ.get('MAX_DRAWDOWN', '10.0'))  # 10%
        self.max_concurrent_trades = int(os.environ.get('MAX_CONCURRENT_TRADES', '5'))
        self.risk_per_trade = 0.02  # 2% per trade
        
    def check_risk_levels(self):
        """Verificar niveles de riesgo actuales"""
        try:
            # Cargar señales actuales
            with open('data/latest_signals.json', 'r') as f:
                signals_data = json.load(f)
            
            signals = signals_data.get('signals', {})
            
            # Verificar número de señales concurrentes
            active_signals = len([s for s in signals.values() if s['signal'] in ['BUY', 'SELL']])
            
            risk_report = {
                'timestamp': datetime.now().isoformat(),
                'active_signals': active_signals,
                'max_allowed': self.max_concurrent_trades,
                'risk_level': 'NORMAL',
                'warnings': [],
                'actions_taken': []
            }
            
            # Verificar límites
            if active_signals > self.max_concurrent_trades:
                risk_report['risk_level'] = 'HIGH'
                risk_report['warnings'].append(f'Too many concurrent signals: {active_signals}')
                
                # Filtrar señales por confianza
                filtered_signals = self.filter_signals_by_confidence(signals)
                risk_report['actions_taken'].append('Filtered signals by confidence')
                
                self.save_filtered_signals(filtered_signals)
            
            # Verificar exposición por par
            self.check_currency_exposure(signals, risk_report)
            
            # Guardar reporte
            self.save_risk_report(risk_report)
            
            return risk_report
            
        except Exception as e:
            print(f"Error checking risk levels: {e}")
            return None
    
    def filter_signals_by_confidence(self, signals):
        """Filtrar señales por confianza cuando hay demasiadas"""
        try:
            # Convertir a lista y ordenar por confianza
            signal_list = [(pair, data) for pair, data in signals.items() if data['signal'] in ['BUY', 'SELL']]
            signal_list.sort(key=lambda x: x[1]['confidence'], reverse=True)
            
            # Tomar solo las mejores señales
            top_signals = signal_list[:self.max_concurrent_trades]
            
            # Crear nuevo diccionario con señales filtradas
            filtered = {}
            for pair, data in top_signals:
                filtered[pair] = data
            
            # Agregar señales HOLD
            for pair, data in signals.items():
                if data['signal'] == 'HOLD':
                    filtered[pair] = data
            
            return filtered
            
        except Exception as e:
            print(f"Error filtering signals: {e}")
            return signals
    
    def check_currency_exposure(self, signals, risk_report):
        """Verificar exposición por moneda"""
        try:
            exposure = {}
            
            for pair, signal_data in signals.items():
                if signal_data['signal'] in ['BUY', 'SELL']:
                    base_currency = pair[:3]
                    quote_currency = pair[3:]
                    
                    # Calcular exposición
                    if signal_data['signal'] == 'BUY':
                        exposure[base_currency] = exposure.get(base_currency, 0) + 1
                        exposure[quote_currency] = exposure.get(quote_currency, 0) - 1
                    else:  # SELL
                        exposure[base_currency] = exposure.get(base_currency, 0) - 1
                        exposure[quote_currency] = exposure.get(quote_currency, 0) + 1
            
            # Verificar exposición excesiva
            max_exposure = 3  # Máximo 3 posiciones en la misma moneda
            
            for currency, exp in exposure.items():
                if abs(exp) > max_exposure:
                    risk_report['warnings'].append(f'High {currency} exposure: {exp}')
                    risk_report['risk_level'] = 'MEDIUM'
            
            risk_report['currency_exposure'] = exposure
            
        except Exception as e:
            print(f"Error checking currency exposure: {e}")
    
    def save_filtered_signals(self, filtered_signals):
        """Guardar señales filtradas"""
        try:
            timestamp = datetime.now().isoformat()
            
            filtered_data = {
                'timestamp': timestamp,
                'signals': filtered_signals,
                'filtered_by': 'risk_manager',
                'reason': 'Too many concurrent signals'
            }
            
            with open('data/latest_signals_filtered.json', 'w') as f:
                json.dump(filtered_data, f, indent=2)
            
            print("Filtered signals saved")
            
        except Exception as e:
            print(f"Error saving filtered signals: {e}")
    
    def save_risk_report(self, risk_report):
        """Guardar reporte de riesgo"""
        try:
            os.makedirs('logs', exist_ok=True)
            
            # Guardar reporte diario
            today = datetime.now().strftime('%Y%m%d')
            risk_file = f"logs/risk_report_{today}.json"
            
            # Cargar reportes existentes
            if os.path.exists(risk_file):
                with open(risk_file, 'r') as f:
                    existing_reports = json.load(f)
            else:
                existing_reports = []
            
            existing_reports.append(risk_report)
            
            with open(risk_file, 'w') as f:
                json.dump(existing_reports, f, indent=2)
            
            # También guardar último reporte
            with open('logs/latest_risk_report.json', 'w') as f:
                json.dump(risk_report, f, indent=2)
            
            print(f"Risk report saved: {risk_report['risk_level']}")
            
        except Exception as e:
            print(f"Error saving risk report: {e}")

if __name__ == "__main__":
    manager = RiskManager()
    report = manager.check_risk_levels()
    
    if report:
        print(f"Risk Level: {report['risk_level']}")
        print(f"Active Signals: {report['active_signals']}")
        if report['warnings']:
            print("Warnings:", report['warnings'])
    
    exit(0 if report and report['risk_level'] != 'HIGH' else 1)
