# scripts/setup_system.py
import os
import json
import subprocess
import sys
from datetime import datetime

class SystemSetup:
    """Automatizar setup inicial del sistema"""
    
    def __init__(self):
        self.setup_log = []
        
    def log(self, message):
        """Log de setup"""
        timestamp = datetime.now().strftime('%H:%M:%S')
        log_entry = f"[{timestamp}] {message}"
        print(log_entry)
        self.setup_log.append(log_entry)
    
    def check_python_dependencies(self):
        """Verificar dependencias de Python"""
        self.log("🔍 Checking Python dependencies...")
        
        required_packages = [
            'pandas', 'numpy', 'scikit-learn', 'requests', 
            'google-auth', 'google-api-python-client'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
                self.log(f"   ✅ {package}")
            except ImportError:
                missing_packages.append(package)
                self.log(f"   ❌ {package}")
        
        if missing_packages:
            self.log(f"📦 Installing missing packages: {missing_packages}")
            
            for package in missing_packages:
                try:
                    subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                    self.log(f"   ✅ Installed {package}")
                except subprocess.CalledProcessError as e:
                    self.log(f"   ❌ Failed to install {package}: {e}")
                    return False
        
        return True
    
    def create_directory_structure(self):
        """Crear estructura de directorios"""
        self.log("📁 Creating directory structure...")
        
        directories = [
            'scripts',
            'config', 
            'models',
            'logs',
            'data',
            'backup',
            '.github/workflows'
        ]
        
        for directory in directories:
            try:
                os.makedirs(directory, exist_ok=True)
                self.log(f"   ✅ Created {directory}")
            except Exception as e:
                self.log(f"   ❌ Failed to create {directory}: {e}")
                return False
        
        return True
    
    def create_env_template(self):
        """Crear template de variables de entorno"""
        self.log("⚙️ Creating environment template...")
        
        env_template = '''# TRADING BOT ENVIRONMENT VARIABLES
# Copy this file to .env and fill in your actual values

# Google Services
GOOGLE_CREDENTIALS={"type":"service_account","project_id":"your-project",...}
GOOGLE_DRIVE_FOLDER=your-google-drive-folder-id
GOOGLE_SPREADSHEET_ID=your-google-spreadsheet-id

# MyFxBook Configuration  
MYFXBOOK_API_KEY=your-myfxbook-api-key
MYFXBOOK_WEBHOOK_URL=your-myfxbook-webhook-url
MYFXBOOK_ACCOUNT_ID=your-myfxbook-account-id
MYFXBOOK_EMAIL=your-myfxbook-email
MYFXBOOK_PASSWORD=your-myfxbook-password

# Risk Management
MAX_DAILY_RISK=5.0
MAX_DRAWDOWN=10.0
MAX_CONCURRENT_TRADES=5

# System Configuration
DEBUG_MODE=false
SIMULATE_TRADES=true
LOG_LEVEL=INFO
'''
        
        try:
            with open('.env.template', 'w') as f:
                f.write(env_template)
            self.log("   ✅ Created .env.template")
            
            # Crear .env si no existe
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write(env_template)
                self.log("   ✅ Created .env (fill in your values)")
            
            return True
            
        except Exception as e:
            self.log(f"   ❌ Failed to create env template: {e}")
            return False
    
    def create_config_files(self):
        """Crear archivos de configuración por defecto"""
        self.log("📝 Creating default configuration files...")
        
        # Trading config
        trading_config = '''# config/local_config.py
# Local configuration overrides

# Override default pairs if needed
CURRENCY_PAIRS = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']

# Override risk settings for your comfort level
RISK_CONFIG = {
    'risk_per_trade': 0.01,  # 1% per trade (conservative)
    'max_daily_risk': 0.02,  # 2% per day (conservative)
    'max_concurrent_trades': 2  # Maximum 2 simultaneous trades
}

# Override ML settings
ML_CONFIG = {
    'confidence_threshold': 0.7,  # Higher threshold = more selective
    'lookback_hours': 500  # Less data = faster training
}
'''
        
        try:
            with open('config/local_config.py', 'w') as f:
                f.write(trading_config)
            self.log("   ✅ Created config/local_config.py")
            return True
            
        except Exception as e:
            self.log(f"   ❌ Failed to create config files: {e}")
            return False
    
    def verify_system_health(self):
        """Verificar salud básica del sistema"""
        self.log("🏥 Verifying system health...")
        
        checks = [
            ('Python version', sys.version_info >= (3, 8)),
            ('Write permissions', os.access('.', os.W_OK)),
            ('Logs directory', os.path.exists('logs')),
            ('Data directory', os.path.exists('data')),
            ('Config directory', os.path.exists('config'))
        ]
        
        all_checks_passed = True
        
        for check_name, check_result in checks:
            if check_result:
                self.log(f"   ✅ {check_name}")
            else:
                self.log(f"   ❌ {check_name}")
                all_checks_passed = False
        
        return all_checks_passed
    
    def create_readme_checklist(self):
        """Crear checklist de setup para el usuario"""
        self.log("📋 Creating setup checklist...")
        
        checklist = '''# TRADING BOT SETUP CHECKLIST

## Automated Setup Completed ✅
- [x] Python dependencies installed
- [x] Directory structure created
- [x] Configuration templates created
- [x] System health verified

## Manual Steps Required ⚠️

### 1. Google Services Setup
- [ ] Create Google Cloud Project
- [ ] Enable Drive and Sheets APIs
- [ ] Create Service Account and download JSON key
- [ ] Create Google Drive folder and get ID
- [ ] Create Google Spreadsheet and get ID
- [ ] Update GOOGLE_* variables in .env

### 2. MyFxBook Setup  
- [ ] Create MyFxBook account
- [ ] Setup AutoTrade
- [ ] Get API key and webhook URL
- [ ] Update MYFXBOOK_* variables in .env

### 3. Admirals Markets Setup
- [ ] Create demo account
- [ ] Connect account to MyFxBook
- [ ] Test copy trading functionality

### 4. GitHub Setup
- [ ] Create GitHub repository
- [ ] Add all secrets in repository settings
- [ ] Commit and push code
- [ ] Verify GitHub Actions run successfully

### 5. Testing
- [ ] Run: python scripts/test_system.py
- [ ] Verify all tests pass
- [ ] Monitor first few trading sessions

### 6. Go Live (After Testing)
- [ ] Start with demo account
- [ ] Monitor performance for 30 days
- [ ] Gradually increase position sizes
- [ ] Never risk more than you can afford to lose

## Important Notes ⚠️
- This is educational software - trading involves significant risk
- Always test thoroughly with demo accounts first
- Never invest money you cannot afford to lose
- Monitor the system regularly for the first weeks
- Have a plan to stop the system if performance degrades

## Support
- Check logs/ directory for error messages
- Run test_system.py to diagnose issues  
- Review GitHub Actions logs for automation issues
'''
        
        try:
            with open('SETUP_CHECKLIST.md', 'w') as f:
                f.write(checklist)
            self.log("   ✅ Created SETUP_CHECKLIST.md")
            return True
            
        except Exception as e:
            self.log(f"   ❌ Failed to create checklist: {e}")
            return False
    
    def run_setup(self):
        """Ejecutar setup completo"""
        self.log("🚀 Starting Trading Bot System Setup...")
        self.log("="*50)
        
        setup_steps = [
            ("Checking Python dependencies", self.check_python_dependencies),
            ("Creating directory structure", self.create_directory_structure),
            ("Creating environment template", self.create_env_template),
            ("Creating config files", self.create_config_files),
            ("Verifying system health", self.verify_system_health),
            ("Creating setup checklist", self.create_readme_checklist)
        ]
        
        failed_steps = []
        
        for step_name, step_function in setup_steps:
            self.log(f"\n🔧 {step_name}...")
            try:
                if not step_function():
                    failed_steps.append(step_name)
            except Exception as e:
                self.log(f"   ❌ Error in {step_name}: {e}")
                failed_steps.append(step_name)
        
        # Guardar log de setup
        self.save_setup_log()
        
        # Resumen final
        self.log("\n" + "="*50)
        if not failed_steps:
            self.log("🎉 SETUP COMPLETED SUCCESSFULLY!")
            self.log("📋 Next: Follow SETUP_CHECKLIST.md for manual steps")
            self.log("🧪 Test: Run 'python scripts/test_system.py'")
        else:
            self.log("⚠️ SETUP COMPLETED WITH WARNINGS")
            self.log(f"❌ Failed steps: {failed_steps}")
            self.log("📋 Review logs and fix issues before proceeding")
        
        self.log("="*50)
        
        return len(failed_steps) == 0
    
    def save_setup_log(self):
        """Guardar log de setup"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'logs/setup_{timestamp}.log'
            
            with open(log_file, 'w') as f:
                f.write('\n'.join(self.setup_log))
            
            # También guardar como latest
            with open('logs/latest_setup.log', 'w') as f:
                f.write('\n'.join(self.setup_log))
            
            self.log(f"📋 Setup log saved to {log_file}")
            
        except Exception as e:
            print(f"Failed to save setup log: {e}")

if __name__ == "__main__":
    setup = SystemSetup()
    success = setup.run_setup()
    exit(0 if success else 1)
