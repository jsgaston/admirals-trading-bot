# scripts/test_system.py
import json
import os
import requests
from datetime import datetime
import pandas as pd
from google_drive_manager import GoogleDriveManager
from ml_signal_generator import MLSignalGenerator

class SystemTester:
    """Testing completo del sistema de trading"""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_status': 'UNKNOWN'
        }
        
    def test_environment_variables(self):
        """Test 1: Verificar variables de entorno"""
        print("🧪 Testing environment variables...")
        
        required_vars = [
            'GOOGLE_CREDENTIALS',
            'GOOGLE_DRIVE_FOLDER', 
            'GOOGLE_SPREADSHEET_ID',
            'MYFXBOOK_API_KEY',
            'MYFXBOOK_WEBHOOK_URL'
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        test_result = {
            'status': 'PASS' if not missing_vars else 'FAIL',
            'missing_variables': missing_vars,
            'message': f'Missing variables: {missing_vars}' if missing_vars else 'All environment variables configured'
        }
        
        self.test_results['tests']['environment_variables'] = test_result
        print(f"   Result: {test_result['status']} - {test_result['message']}")
        
        return test_result['status'] == 'PASS'
    
    def test_google_drive_connection(self):
        """Test 2: Conexión con Google Drive"""
        print("🧪 Testing Google Drive connection...")
        
        try:
            drive_manager = GoogleDriveManager()
            
            if drive_manager.service is None:
                test_result = {
                    'status': 'FAIL',
                    'message': 'Could not authenticate with Google Drive'
                }
            else:
                # Test upload simple
                test_data = {'test': 'data', 'timestamp': datetime.now().isoformat()}
                success = drive_manager.upload_signals({'signals': {}, 'test_data': test_data})
                
                test_result = {
                    'status': 'PASS' if success else 'FAIL',
                    'message': 'Google Drive upload successful' if success else 'Google Drive upload failed'
                }
            
        except Exception as e:
            test_result = {
                'status': 'FAIL',
                'message': f'Google Drive test error: {str(e)}'
            }
        
        self.test_results['tests']['google_drive'] = test_result
        print(f"   Result: {test_result['status']} - {test_result['message']}")
        
        return test_result['status'] == 'PASS'
    
    def test_data_collection(self):
        """Test 3: Recolección de datos forex"""
        print("🧪 Testing forex data collection...")
        
        try:
            from data_collector import ForexDataCollector
            
            collector = ForexDataCollector()
            
            # Test obtener tasas actuales
            rates = collector.get_current_rates()
            
            if rates and len(rates) > 0:
                # Test calcular tasa de par
                eurusd_rate = collector.calculate_pair_rate('EURUSD', rates)
                
                if eurusd_rate:
                    test_result = {
                        'status': 'PASS',
                        'message': f'Data collection successful, EUR/USD: {eurusd_rate}',
                        'rates_count': len(rates),
                        'sample_rate': eurusd_rate
                    }
                else:
                    test_result = {
                        'status': 'FAIL',
                        'message': 'Could not calculate EUR/USD rate'
                    }
            else:
                test_result = {
                    'status': 'FAIL',
                    'message': 'No forex rates obtained'
                }
                
        except Exception as e:
            test_result = {
                'status': 'FAIL',
                'message': f'Data collection error: {str(e)}'
            }
        
        self.test_results['tests']['data_collection'] = test_result
        print(f"   Result: {test_result['status']} - {test_result['message']}")
        
        return test_result['status'] == 'PASS'
    
    def test_ml_model_training(self):
        """Test 4: Entrenamiento del modelo ML"""
        print("🧪 Testing ML model training...")
        
        try:
            generator = MLSignalGenerator()
            
            # Generar datos sintéticos para testing
            test_data = []
            base_price = 1.1800
            
            for i in range(200):  # 200 puntos de datos
                price = base_price * (1 + np.random.normal(0, 0.001))
                test_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'open': price,
                    'high': price * 1.001,
                    'low': price * 0.999,
                    'close': price,
                    'volume': 1000
                })
            
            df = pd.DataFrame(test_data)
            
            # Test cálculo de indicadores técnicos
            df_with_indicators = generator.calculate_technical_indicators(df)
            
            if len(df_with_indicators) > 50:
                test_result = {
                    'status': 'PASS',
                    'message': f'ML model training test successful, {len(df_with_indicators)} data points processed',
                    'indicators_calculated': len([col for col in df_with_indicators.columns if col not in test_data[0].keys()])
                }
            else:
                test_result = {
                    'status': 'FAIL',
                    'message': 'Insufficient data after indicator calculation'
                }
                
        except Exception as e:
            test_result = {
                'status': 'FAIL',
                'message': f'ML model test error: {str(e)}'
            }
        
        self.test_results['tests']['ml_model'] = test_result
        print(f"   Result: {test_result['status']} - {test_result['message']}")
        
        return test_result['status'] == 'PASS'
    
    def test_myfxbook_connection(self):
        """Test 5: Conexión con MyFxBook"""
        print("🧪 Testing MyFxBook connection...")
        
        try:
            webhook_url = os.environ.get('MYFXBOOK_WEBHOOK_URL')
            
            if not webhook_url or 'your-' in webhook_url:
                test_result = {
                    'status': 'SKIP',
                    'message': 'MyFxBook webhook URL not configured'
                }
            else:
                # Test webhook con datos de prueba
                test_payload = {
                    'symbol': 'EURUSD',
                    'action': 'buy',
                    'volume': 0.01,
                    'comment': 'System_Test',
                    'timestamp': datetime.now().isoformat()
                }
                
                response = requests.post(webhook_url, json=test_payload, timeout=10)
                
                if response.status_code == 200:
                    test_result = {
                        'status': 'PASS',
                        'message': 'MyFxBook webhook test successful'
                    }
                else:
                    test_result = {
                        'status': 'FAIL',
                        'message': f'MyFxBook webhook failed: HTTP {response.status_code}'
                    }
                    
        except Exception as e:
            test_result = {
                'status': 'FAIL',
                'message': f'MyFxBook test error: {str(e)}'
            }
        
        self.test_results['tests']['myfxbook'] = test_result
        print(f"   Result: {test_result['status']} - {test_result['message']}")
        
        return test_result['status'] == 'PASS'
    
    def test_risk_management(self):
        """Test 6: Sistema de gestión de riesgo"""
        print("🧪 Testing risk management...")
        
        try:
            from risk_manager import RiskManager
            
            manager = RiskManager()
            
            # Crear señales de prueba
            test_signals = {
                'timestamp': datetime.now().isoformat(),
                'signals': {}
            }
            
            # Generar muchas señales para probar límites
            pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCHF', 'EURGBP']
            for pair in pairs:
                test_signals['signals'][pair] = {
                    'signal': 'BUY',
                    'confidence': 0.8,
                    'position_size': 0.1
                }
            
            # Guardar señales temporales para testing
            os.makedirs('data', exist_ok=True)
            with open('data/latest_signals.json', 'w') as f:
                json.dump(test_signals, f)
            
            # Test risk check
            risk_report = manager.check_risk_levels()
            
            if risk_report and 'risk_level' in risk_report:
                test_result = {
                    'status': 'PASS',
                    'message': f'Risk management test successful, risk level: {risk_report["risk_level"]}',
                    'active_signals': risk_report.get('active_signals', 0),
                    'risk_level': risk_report['risk_level']
                }
            else:
                test_result = {
                    'status': 'FAIL',
                    'message': 'Risk management test failed'
                }
                
        except Exception as e:
            test_result = {
                'status': 'FAIL',
                'message': f'Risk management error: {str(e)}'
            }
        
        self.test_results['tests']['risk_management'] = test_result
        print(f"   Result: {test_result['status']} - {test_result['message']}")
        
        return test_result['status'] == 'PASS'
    
    def run_all_tests(self):
        """Ejecutar todos los tests"""
        print("🔬 Starting system testing...\n")
        
        tests = [
            self.test_environment_variables,
            self.test_google_drive_connection,
            self.test_data_collection,
            self.test_ml_model_training,
            self.test_myfxbook_connection,
            self.test_risk_management
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test in tests:
            try:
                if test():
                    passed_tests += 1
            except Exception as e:
                print(f"   Test error: {e}")
            print()  # Empty line between tests
        
        # Calcular resultado general
        pass_rate = (passed_tests / total_tests) * 100
        
        if pass_rate >= 80:
            overall_status = 'PASS'
        elif pass_rate >= 60:
            overall_status = 'WARNING'
        else:
            overall_status = 'FAIL'
        
        self.test_results['overall_status'] = overall_status
        self.test_results['pass_rate'] = pass_rate
        self.test_results['tests_passed'] = passed_tests
        self.test_results['total_tests'] = total_tests
        
        # Guardar resultados
        self.save_test_results()
        
        # Resumen final
        print("="*50)
        print(f"🔬 SYSTEM TESTING COMPLETED")
        print(f"Overall Status: {overall_status}")
        print(f"Pass Rate: {pass_rate:.1f}% ({passed_tests}/{total_tests})")
        print("="*50)
        
        return overall_status == 'PASS'
    
    def save_test_results(self):
        """Guardar resultados de tests"""
        try:
            os.makedirs('logs', exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'logs/system_test_{timestamp}.json'
            
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            
            # También guardar como latest
            with open('logs/latest_system_test.json', 'w') as f:
                json.dump(self.test_results, f, indent=2)
            
            print(f"📋 Test results saved to {filename}")
            
        except Exception as e:
            print(f"Error saving test results: {e}")

# Import numpy for testing
import numpy as np

if __name__ == "__main__":
    tester = SystemTester()
    success = tester.run_all_tests()
    exit(0 if success else 1)
