# config/trading_config.py
import os
from datetime import datetime

class TradingConfig:
    """Configuración principal del sistema de trading"""
    
    # Pares de divisas para trading
    CURRENCY_PAIRS = [
        'EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 
        'USDCHF', 'EURGBP', 'EURJPY', 'GBPJPY'
    ]
    
    # Configuración de Machine Learning
    ML_CONFIG = {
        'model_type': 'RandomForest',
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_split': 5,
        'confidence_threshold': 0.6,
        'lookback_hours': 720,  # 30 días
        'forward_prediction_hours': 4
    }
    
    # Configuración de riesgo
    RISK_CONFIG = {
        'risk_per_trade': 0.02,  # 2% del capital por trade
        'max_daily_risk': 0.05,  # 5% del capital por día
        'max_drawdown': 0.10,    # 10% máximo drawdown
        'max_concurrent_trades': 5,
        'max_currency_exposure': 3  # Máximo 3 posiciones por moneda
    }
    
    # Configuración de APIs
    API_CONFIG = {
        'exchangerate_api': 'https://api.exchangerate-api.com/v4',
        'request_timeout': 30,
        'retry_attempts': 3,
        'rate_limit_delay': 1  # segundos entre requests
    }
    
    # Configuración de timeframes
    TIMEFRAMES = {
        'data_collection': '1H',
        'signal_generation': '1H',
        'risk_management': '15M'
    }
    
    # Configuración de Google Services
    GOOGLE_CONFIG = {
        'drive_folder_id': os.environ.get('GOOGLE_DRIVE_FOLDER'),
        'spreadsheet_id': os.environ.get('GOOGLE_SPREADSHEET_ID'),
        'credentials_path': 'GOOGLE_CREDENTIALS'  # Environment variable
    }
    
    # Configuración de MyFxBook
    MYFXBOOK_CONFIG = {
        'api_key': os.environ.get('MYFXBOOK_API_KEY'),
        'webhook_url': os.environ.get('MYFXBOOK_WEBHOOK_URL'),
        'account_id': os.environ.get('MYFXBOOK_ACCOUNT_ID'),
        'email': os.environ.get('MYFXBOOK_EMAIL'),
        'password': os.environ.get('MYFXBOOK_PASSWORD')
    }
    
    # Configuración de logging
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        'log_to_file': True,
        'log_retention_days': 30
    }
    
    # Configuración de scheduler
    SCHEDULER_CONFIG = {
        'data_collection_interval': '0 */6 * * *',  # Cada 6 horas
        'signal_generation_interval': '30 */4 * * *',  # Cada 4 horas
        'risk_check_interval': '*/15 * * * *',  # Cada 15 minutos
        'performance_update_interval': '0 0 * * *'  # Diario a medianoche
    }
    
    @classmethod
    def validate_config(cls):
        """Validar configuración"""
        required_env_vars = [
            'GOOGLE_DRIVE_FOLDER',
            'GOOGLE_SPREADSHEET_ID', 
            'GOOGLE_CREDENTIALS',
            'MYFXBOOK_API_KEY',
            'MYFXBOOK_WEBHOOK_URL'
        ]
        
        missing_vars = []
        for var in required_env_vars:
            if not os.environ.get(var):
                missing_vars.append(var)
        
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {missing_vars}")
        
        return True
    
    @classmethod
    def get_runtime_config(cls):
        """Obtener configuración para runtime"""
        return {
            'pairs': cls.CURRENCY_PAIRS,
            'ml_config': cls.ML_CONFIG,
            'risk_config': cls.RISK_CONFIG,
            'api_config': cls.API_CONFIG,
            'timeframes': cls.TIMEFRAMES
        }
