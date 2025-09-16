Arquitectura Final: 100% Gratuita

trading-bot-system/
├── .github/workflows/
│   └── trading-system.yml     # Workflow principal
├── scripts/
│   ├── data_collector.py      # Recolección datos
│   ├── entry_signal_generator.py  # Señales ML
│   ├── risk_calculator.py     # Stop Loss dinámico
│   └── risk_executor.py       # Ejecución risk mgmt
├── models/
│   └── trading_model.py       # Modelo DL
├── config/
│   └── trading_config.py      # Configuración
└── data/                      # Datos temporales


timeframe_strategy = {
    'Entry_Signals': {
        'timeframe': '1H',
        'lookback': '720 points (30 days)',
        'frequency': 'Every 4 hours',
        'purpose': 'Deep Learning predictions'
    },
    
    'Risk_Management': {
        'timeframe': '15M', 
        'lookback': '96 points (24 hours)',
        'frequency': 'Every 15 minutes',
        'purpose': 'Dynamic stop loss adjustment'
    },
    
    'Position_Monitoring': {
        'timeframe': '5M',
        'lookback': '288 points (24 hours)', 
        'frequency': 'Real-time via webhook',
        'purpose': 'Immediate exit decisions'
    }
}

