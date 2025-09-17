# config/broker_config.py

class BrokerConfig:
    """Configuración específica del broker"""
    
    # Configuración para Admirals Markets
    ADMIRALS_CONFIG = {
        'name': 'Admirals Markets',
        'platform': 'MT5',
        'demo_server': 'AdmiralsGroup-Demo',
        'live_server': 'AdmiralsGroup-Live',
        'max_slippage': 3,  # pips
        'default_lots': 0.01,
        'min_lots': 0.01,
        'max_lots': 100.0,
        'pip_values': {
            'EURUSD': 10,    # USD per pip for 1 lot
            'GBPUSD': 10,
            'USDJPY': 10,
            'AUDUSD': 10,
            'USDCHF': 10,
            'EURGBP': 10,
            'EURJPY': 10,
            'GBPJPY': 10
        },
        'typical_spreads': {  # pips
            'EURUSD': 0.6,
            'GBPUSD': 0.9,
            'USDJPY': 0.7,
            'AUDUSD': 0.9,
            'USDCHF': 1.1,
            'EURGBP': 0.8,
            'EURJPY': 1.2,
            'GBPJPY': 1.5
        }
    }
    
    # Configuración para MyFxBook AutoTrade
    MYFXBOOK_AUTOTRADE_CONFIG = {
        'signal_provider_name': 'ML_Trading_Bot',
        'signal_description': 'Machine Learning Forex Trading Signals',
        'subscription_fee': 0,  # Gratis inicialmente
        'max_subscribers': 100,
        'min_balance_required': 1000,  # USD
        'risk_warning': 'Past performance is not indicative of future results'
    }
    
    # Configuración de ejecución
    EXECUTION_CONFIG = {
        'order_type': 'MARKET',
        'execution_mode': 'INSTANT',
        'slippage_tolerance': 3,  # pips
        'partial_fill': True,
        'stop_loss_type': 'DYNAMIC',  # FIXED or DYNAMIC
        'take_profit_type': 'DYNAMIC',
        'trailing_stop': True,
        'trailing_step': 5  # pips
    }
    
    # Configuración de riesgo por broker
    BROKER_RISK_CONFIG = {
        'admirals_demo': {
            'max_position_size': 1.0,   # lots
            'max_daily_trades': 20,
            'max_exposure_per_pair': 2.0,  # lots
            'account_balance': 10000,   # USD demo
            'leverage': 1,  # Sin leverage para demo
            'currency_base': 'USD'
        },
        'admirals_live': {
            'max_position_size': 0.1,   # lots - conservador para live
            'max_daily_trades': 10,
            'max_exposure_per_pair': 0.2,  # lots
            'account_balance': 1000,    # USD mínimo recomendado
            'leverage': 30,             # Máximo 1:30 en EU
            'currency_base': 'EUR'
        }
    }
    
    @classmethod
    def get_broker_config(cls, broker_name='admirals', account_type='demo'):
        """Obtener configuración del broker"""
        if broker_name.lower() == 'admirals':
            config = cls.ADMIRALS_CONFIG.copy()
            config.update(cls.EXECUTION_CONFIG)
            
            risk_key = f'admirals_{account_type}'
            if risk_key in cls.BROKER_RISK_CONFIG:
                config['risk'] = cls.BROKER_RISK_CONFIG[risk_key]
            
            return config
        
        raise ValueError(f"Unknown broker: {broker_name}")
    
    @classmethod
    def calculate_position_size(cls, pair, account_balance, risk_percent, stop_loss_pips, broker='admirals'):
        """Calcular tamaño de posición basado en riesgo"""
        try:
            risk_amount = account_balance * (risk_percent / 100)
            
            config = cls.get_broker_config(broker)
            pip_value = config['pip_values'].get(pair, 10)
            
            # Calcular lots
            stop_loss_amount = stop_loss_pips * pip_value
            
            if stop_loss_amount <= 0:
                return config['risk']['max_position_size']
            
            position_size = risk_amount / stop_loss_amount
            
            # Aplicar límites
            min_lot = 0.01
            max_lot = config['risk']['max_position_size']
            
            return max(min_lot, min(max_lot, round(position_size, 2)))
            
        except Exception as e:
            print(f"Error calculating position size: {e}")
            return 0.01
