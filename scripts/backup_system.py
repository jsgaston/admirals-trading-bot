# scripts/backup_system.py
import os
import json
import shutil
import zipfile
from datetime import datetime, timedelta
import subprocess

class SystemBackup:
    """Backup y recovery del sistema de trading"""
    
    def __init__(self):
        self.backup_dir = 'backup'
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
    def create_full_backup(self):
        """Crear backup completo del sistema"""
        print(f"💾 Creating full system backup...")
        
        try:
            # Crear directorio de backup
            backup_path = os.path.join(self.backup_dir, f'full_backup_{self.timestamp}')
            os.makedirs(backup_path, exist_ok=True)
            
            # Directorios a respaldar
            directories_to_backup = [
                'scripts',
                'config', 
                'models',
                'logs',
                'data',
                '.github'
            ]
            
            # Archivos individuales a respaldar
            files_to_backup = [
                'requirements.txt',
                '.env',
                '.env.template',
                'README.md',
                'SETUP_CHECKLIST.md'
            ]
            
            # Copiar directorios
            for directory in directories_to_backup:
                if os.path.exists(directory):
                    dest_dir = os.path.join(backup_path, directory)
                    shutil.copytree(directory, dest_dir, dirs_exist_ok=True)
                    print(f"   ✅ Backed up {directory}")
            
            # Copiar archivos
            for file in files_to_backup:
                if os.path.exists(file):
                    shutil.copy2(file, backup_path)
                    print(f"   ✅ Backed up {file}")
            
            # Crear metadata de backup
            metadata = {
                'backup_timestamp': self.timestamp,
                'backup_type': 'full',
                'directories': directories_to_backup,
                'files': files_to_backup,
                'git_commit': self._get_git_commit(),
                'python_version': subprocess.check_output([sys.executable, '--version']).decode().strip()
            }
            
            with open(os.path.join(backup_path, 'backup_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Crear ZIP del backup
            zip_path = f'{backup_path}.zip'
            self._create_zip(backup_path, zip_path)
            
            # Limpiar directorio temporal
            shutil.rmtree(backup_path)
            
            print(f"✅ Full backup created: {zip_path}")
            return zip_path
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
            return None
    
    def create_data_backup(self):
        """Crear backup solo de datos"""
        print(f"💾 Creating data backup...")
        
        try:
            backup_path = os.path.join(self.backup_dir, f'data_backup_{self.timestamp}')
            os.makedirs(backup_path, exist_ok=True)
            
            # Copiar directorios de datos
            data_dirs = ['logs', 'data', 'models']
            
            for directory in data_dirs:
                if os.path.exists(directory):
                    dest_dir = os.path.join(backup_path, directory)
                    shutil.copytree(directory, dest_dir, dirs_exist_ok=True)
                    print(f"   ✅ Backed up {directory}")
            
            # Metadata
            metadata = {
                'backup_timestamp': self.timestamp,
                'backup_type': 'data_only',
                'directories': data_dirs
            }
            
            with open(os.path.join(backup_path, 'backup_metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            # Crear ZIP
            zip_path = f'{backup_path}.zip'
            self._create_zip(backup_path, zip_path)
            shutil.rmtree(backup_path)
            
            print(f"✅ Data backup created: {zip_path}")
            return zip_path
            
        except Exception as e:
            print(f"❌ Data backup failed: {e}")
            return None
    
    def restore_backup(self, backup_file):
        """Restaurar desde backup"""
        print(f"🔄 Restoring from backup: {backup_file}")
        
        try:
            if not os.path.exists(backup_file):
                print(f"❌ Backup file not found: {backup_file}")
                return False
            
            # Crear directorio temporal para extracción
            extract_path = f'temp_restore_{self.timestamp}'
            
            with zipfile.ZipFile(backup_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            # Leer metadata
            metadata_path = os.path.join(extract_path, 'backup_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                print(f"   📋 Backup from: {metadata.get('backup_timestamp')}")
                print(f"   📋 Type: {metadata.get('backup_type')}")
            
            # Confirmar restauración
            response = input("⚠️ This will overwrite current files. Continue? (y/N): ")
            if response.lower() != 'y':
                shutil.rmtree(extract_path)
                print("❌ Restore cancelled")
                return False
            
            # Copiar archivos de vuelta
            for item in os.listdir(extract_path):
                if item == 'backup_metadata.json':
                    continue
                    
                source_path = os.path.join(extract_path, item)
                dest_path = item
                
                if os.path.isdir(source_path):
                    if os.path.exists(dest_path):
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_path, dest_path)
                    print(f"   ✅ Restored directory: {item}")
                else:
                    shutil.copy2(source_path, dest_path)
                    print(f"   ✅ Restored file: {item}")
            
            # Limpiar directorio temporal
            shutil.rmtree(extract_path)
            
            print("✅ Restore completed successfully")
            return True
            
        except Exception as e:
            print(f"❌ Restore failed: {e}")
            return False
    
    def cleanup_old_backups(self, keep_days=30):
        """Limpiar backups antiguos"""
        print(f"🧹 Cleaning up backups older than {keep_days} days...")
        
        try:
            if not os.path.exists(self.backup_dir):
                return
            
            cutoff_date = datetime.now() - timedelta(days=keep_days)
            removed_count = 0
            
            for file in os.listdir(self.backup_dir):
                file_path = os.path.join(self.backup_dir, file)
                
                if os.path.isfile(file_path) and file.endswith('.zip'):
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    if file_time < cutoff_date:
                        os.remove(file_path)
                        print(f"   🗑️ Removed: {file}")
                        removed_count += 1
            
            print(f"✅ Cleaned up {removed_count} old backups")
            
        except Exception as e:
            print(f"❌ Cleanup failed: {e}")
    
    def _create_zip(self, source_dir, zip_path):
        """Crear archivo ZIP"""
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
    
    def _get_git_commit(self):
        """Obtener commit actual de git"""
        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode().strip()
            return commit
        except:
            return 'unknown'
    
    def list_backups(self):
        """Listar backups disponibles"""
        print("📋 Available backups:")
        
        if not os.path.exists(self.backup_dir):
            print("   No backups found")
            return []
        
        backups = []
        
        for file in os.listdir(self.backup_dir):
            if file.endswith('.zip'):
                file_path = os.path.join(self.backup_dir, file)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                file_size = os.path.getsize(file_path) / (1024 * 1024)  # MB
                
                backups.append({
                    'filename': file,
                    'path': file_path,
                    'timestamp': file_time,
                    'size_mb': file_size
                })
        
        # Ordenar por fecha
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for backup in backups:
            print(f"   📦 {backup['filename']}")
            print(f"      Date: {backup['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"      Size: {backup['size_mb']:.1f} MB")
            print()
        
        return backups

import sys

if __name__ == "__main__":
    backup_system = SystemBackup()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python backup_system.py full          # Create full backup")
        print("  python backup_system.py data          # Create data backup")
        print("  python backup_system.py restore <file> # Restore from backup")
        print("  python backup_system.py list          # List available backups")
        print("  python backup_system.py cleanup       # Clean old backups")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == 'full':
        backup_system.create_full_backup()
    elif command == 'data':
        backup_system.create_data_backup()
    elif command == 'restore' and len(sys.argv) == 3:
        backup_system.restore_backup(sys.argv[2])
    elif command == 'list':
        backup_system.list_backups()
    elif command == 'cleanup':
        backup_system.cleanup_old_backups()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
