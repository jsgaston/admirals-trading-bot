# Trading Bot Automatizado - Admirals + MyFxBook

Sistema completo de trading algorítmico que utiliza Machine Learning para generar señales de forex y las ejecuta automáticamente a través de MyFxBook AutoTrade conectado a Admirals Markets.

## 🏗️ Arquitectura del Sistema

```
GitHub Actions → ML Signals → Google Drive → MyFxBook AutoTrade → Admirals Markets
     ↓              ↓             ↓              ↓                    ↓
Performance ← Google Sheets ← Apps Script ← Trading Logs ← Trade Execution
```

## 🚀 Características

- **Machine Learning**: Modelos RandomForest con 20+ indicadores técnicos
- **Automatización Completa**: Sin intervención manual necesaria
- **Risk Management**: Control de riesgo avanzado con múltiples parámetros
- **Monitoreo en Tiempo Real**: Dashboard en Google Sheets
- **Multiple Timeframes**: 1H para señales, 15M para risk management
- **Performance Tracking**: Análisis detallado de resultados
- **Notificaciones**: Alertas por email para eventos importantes

## 📋 Requisitos Previos

### Cuentas Necesarias (Todas Gratuitas)
1. **GitHub** - Para hosting del código y GitHub Actions
2. **Google Account** - Para Drive, Sheets y Apps Script
3. **MyFxBook** - Para AutoTrade y copy trading
4. **Admirals Markets** - Cuenta demo para testing

### APIs y Keys Necesarias
- Google Service Account (gratuito)
- MyFxBook API Key (gratuito)
- ExchangeRate API (gratuito, 1500 calls/mes)

## 🛠️ Instalación Paso a Paso

### Paso 1: Configurar GitHub Repository

```bash
# 1. Crear repositorio en GitHub
# Ir a github.com/new
# Nombre: admirals-trading-bot
# Public/Private según preferencia

# 2. Clonar localmente
git clone https://github.com/tu-usuario/admirals-trading-bot.git
cd admirals-trading-bot

# 3. Crear estructura de directorios
mkdir -p .github/workflows scripts config models google_apps_script logs data

# 4. Copiar todos los archivos del sistema a sus respectivos directorios
```

### Paso 2: Configurar Google Services

#### 2.1 Crear Google Service Account
```bash
# 1. Ir a Google Cloud Console: console.cloud.google.com
# 2. Crear nuevo proyecto: "trading-bot-project"
# 3. Habilitar APIs:
#    - Google Drive API
#    - Google Sheets API
# 4. Crear Service Account:
#    - IAM & Admin > Service Accounts
#    - Create Service Account: "trading-bot-sa"
#    - Create Key (JSON) y descargar
```

#### 2.2 Configurar Google Drive
```bash
# 1. Crear carpeta en Google Drive: "Trading Bot Data"
# 2. Compartir carpeta con service account email
# 3. Copiar Folder ID desde URL
```

#### 2.3 Crear Google Spreadsheet
```bash
# 1. Crear nuevo Google Sheets: "Trading Bot Dashboard"
# 2. Compartir con service account email
# 3. Copiar Spreadsheet ID desde URL
```

### Paso 3: Configurar MyFxBook

#### 3.1 Crear Cuenta MyFxBook
```bash
# 1. Registrarse en myfxbook.com
# 2. Verificar email
# 3. Conectar cuenta demo de Admirals
```

#### 3.2 Configurar AutoTrade
```bash
# 1. Ir a MyFxBook > AutoTrade
# 2. Create Signal Provider
# 3. Nombre: "ML Trading Bot"
# 4. Configurar webhooks
# 5. Obtener API Key desde Settings
```

### Paso 4: Configurar Admirals Markets

#### 4.1 Cuenta Demo
```bash
# 1. Ir a admirals.com
# 2. Abrir cuenta demo
# 3. Descargar MT5
# 4. Conectar cuenta demo
```

#### 4.2 Conectar con MyFxBook
```bash
# 1. En MyFxBook: Add Account
# 2. Seleccionar Admirals Markets
# 3. Introducir credenciales de cuenta demo
# 4. Verificar conexión
```

### Paso 5: Configurar GitHub Secrets

```bash
# En GitHub Repository: Settings > Secrets and variables > Actions

# Agregar estos secrets:
GOOGLE_CREDENTIALS={"type":"service_account",...}  # JSON completo del service account
GOOGLE_DRIVE_FOLDER=1AbC...  # ID de la carpeta de Drive
GOOGLE_SPREADSHEET_ID=1XyZ...  # ID del spreadsheet
MYFXBOOK_API_KEY=your_api_key
MYFXBOOK_WEBHOOK_URL=https://www.myfxbook.com/api/webhook/your_id
MYFXBOOK_ACCOUNT_ID=your_account_id
MYFXBOOK_EMAIL=your_myfxbook_email
MYFXBOOK_PASSWORD=your_myfxbook_password
MAX_DAILY_RISK=5.0
MAX_DRAWDOWN=10.0
MAX_CONCURRENT_TRADES=5
```

### Paso 6: Configurar Google Apps Script

#### 6.1 Crear Apps Script Project
```bash
# 1. Ir a script.google.com
# 2. New Project
# 3. Nombre: "Trading Bot Monitor"
# 4. Copiar código de google_apps_script/Code.gs
```

#### 6.2 Configurar Triggers
```bash
# 1. En Apps Script: Triggers (icono reloj)
# 2. Add Trigger:
#    - Function: mainTradingUpdate
#    - Event source: Time-driven
#    - Type: Hour timer
#    - Hour interval: Every hour
```

### Paso 7: Testing Inicial

#### 7.1 Test GitHub Actions
```bash
# 1. Hacer commit y push del código
git add .
git commit -m "Initial trading bot setup"
git push origin main

# 2. Ir a GitHub > Actions
# 3. Verificar que workflows aparecen
# 4. Ejecutar manualmente "Data Collection"
```

#### 7.2 Test Google Services
```bash
# 1. Verificar que archivos aparecen en Google Drive
# 2. Verificar que Google Sheets se actualiza
# 3. Verificar logs en Apps Script
```

#### 7.3 Test MyFxBook Integration
```bash
# 1. Verificar que señales llegan a MyFxBook
# 2. Verificar conexión con Admirals
# 3. Verificar que copy trading funciona
```

## ⚙️ Configuración Avanzada

### Ajustar Parámetros de Trading

```python
# Editar config/trading_config.py

# Pares de trading
CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

# Risk management
RISK_CONFIG = {
    'risk_per_trade': 0.01,  # 1% más conservador
    'max_daily_risk': 0.03,  # 3% por día
    'max_concurrent_trades': 3  # Menos trades simultáneos
}

# ML Model
ML_CONFIG = {
    'confidence_threshold': 0.7,  # Más selectivo
    'lookback_hours': 1000  # Más datos históricos
}
```

### Configurar Notificaciones

```python
# En Google Apps Script, configurar email alerts
const NOTIFICATION_EMAIL = 'tu-email@gmail.com';

# Tipos de alertas:
# - Señales de alta confianza (>90%)
# - Muchas señales simultáneas (>5)
# - Errores del sistema
# - Performance diaria
```

## 📊 Monitoreo y Mantenimiento

### Dashboard en Google Sheets

El spreadsheet contiene 3 hojas:

1. **Current Signals**: Señales activas
2. **Trading History**: Historial de todas las señales
3. **Dashboard**: Métricas y estadísticas

### Logs y Debugging

```bash
# Logs de GitHub Actions
# GitHub > Actions > Select workflow > View logs

# Logs de Google Apps Script
# script.google.com > Select project > Executions

# Logs locales (si corres scripts manualmente)
tail -f logs/collector.log
tail -f logs/risk_report_20240101.json
```

### Performance Monitoring

```python
# Verificar performance regularmente
python scripts/performance_tracker.py

# Métricas clave a monitorear:
# - Win rate > 60%
# - Profit factor > 1.5
# - Max drawdown < 10%
# - Sharpe ratio > 1.0
```

## 🔧 Troubleshooting

### Problemas Comunes

#### GitHub Actions no ejecuta
```bash
# Verificar:
# 1. Syntax de YAML workflows
# 2. Secrets configurados correctamente
# 3. Repository permissions
# 4. Billing de GitHub Actions (debería ser gratis)
```

#### Google Drive/Sheets no actualiza
```bash
# Verificar:
# 1. Service account tiene permisos
# 2. IDs correctos en secrets
# 3. APIs habilitadas en Google Cloud
# 4. Quota de API no excedida
```

#### MyFxBook no recibe señales
```bash
# Verificar:
# 1. Webhook URL correcta
# 2. API Key válida
# 3. Account conectada
# 4. AutoTrade activado
```

#### Admirals no ejecuta trades
```bash
# Verificar:
# 1. Cuenta demo activa
# 2. MT5 corriendo (si usas EA)
# 3. Copy trading habilitado en MyFxBook
# 4. Balance suficiente
```

### Debug Mode

```bash
# Para testing intensivo, habilitar debug mode:
# En config/trading_config.py
DEBUG_MODE = True
SIMULATE_TRADES = True  # No enviar órdenes reales
LOG_LEVEL = 'DEBUG'
```

## 📈 Optimización de Performance

### Backtesting

```python
# Ejecutar backtest histórico
python scripts/backtesting.py --start-date 2024-01-01 --end-date 2024-12-01

# Métricas esperadas para un buen modelo:
# Win Rate: 60-70%
# Profit Factor: 1.5-2.5
# Max Drawdown: <15%
# Sharpe Ratio: >1.2
```

### Ajuste de Hiperparámetros

```python
# Optimizar ML model
# En scripts/ml_signal_generator.py

# Probar diferentes configuraciones:
# - n_estimators: [100, 200, 300]
# - max_depth: [8, 10, 12]
# - confidence_threshold: [0.6, 0.7, 0.8]
# - lookback_hours: [500, 720, 1000]
```

## 🚨 Consideraciones de Riesgo

### Trading Real vs Demo

**IMPORTANTE**: El sistema está configurado para trading demo por defecto.

Para trading real:
1. Usar cuentas pequeñas inicialmente
2. Monitorear performance por semanas antes de aumentar capital
3. Nunca usar más del 1-2% del capital por trade
4. Tener plan de contingencia si el sistema falla

### Risk Management

```python
# Parámetros conservadores recomendados:
RISK_CONFIG = {
    'risk_per_trade': 0.01,      # 1% máximo por trade
    'max_daily_risk': 0.02,      # 2% máximo por día
    'max_drawdown': 0.05,        # 5% stop trading
    'max_concurrent_trades': 2   # Máximo 2 trades simultáneos
}
```

## 📚 Resources Adicionales

### Documentación APIs
- [GitHub Actions](https://docs.github.com/en/actions)
- [Google Drive API](https://developers.google.com/drive/api)
- [Google Sheets API](https://developers.google.com/sheets/api)
- [MyFxBook API](https://www.myfxbook.com/api)

### Trading Resources
- [Admirals Education](https://admirals.com/education)
- [MyFxBook Community](https://www.myfxbook.com/community)
- [MT5 Documentation](https://www.metatrader5.com/en/terminal/help)

## 🤝 Support

### Community
- GitHub Issues para bugs y feature requests
- Discord/Telegram community (crear si hay interés)

### Professional Support
- Consultation para optimización de parámetros
- Custom development para features específicas
- Portfolio management advice

## 📄 License

MIT License - Ver LICENSE file para detalles.

## ⚠️ Disclaimer

Este software es para propósitos educacionales y de investigación. Trading forex conlleva riesgo significativo de pérdida. Nunca inviertas dinero que no puedas permitirte perder. Performance pasada no garantiza resultados futuros.

---

**Happy Trading! 📈**
