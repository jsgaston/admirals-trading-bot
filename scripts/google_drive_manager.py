# scripts/google_drive_manager.py
import os
import json
import io
from datetime import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload, MediaIoBaseDownload

class GoogleDriveManager:
    def __init__(self):
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.folder_id = os.environ.get('GOOGLE_DRIVE_FOLDER')
        self.service = self._authenticate()
        
    def _authenticate(self):
        """Autenticar con Google Drive"""
        try:
            credentials_json = os.environ.get('GOOGLE_CREDENTIALS')
            
            if not credentials_json:
                print("Google credentials not found")
                return None
            
            credentials_info = json.loads(credentials_json)
            credentials = Credentials.from_service_account_info(
                credentials_info,
                scopes=self.scopes
            )
            
            return build('drive', 'v3', credentials=credentials)
            
        except Exception as e:
            print(f"Error authenticating Google Drive: {e}")
            return None
    
    def upload_forex_data(self, data):
        """Subir datos forex a Google Drive"""
        try:
            if not self.service:
                return False
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            for pair, pair_data in data.items():
                # Preparar archivo JSON
                file_content = {
                    'pair': pair,
                    'timestamp': datetime.now().isoformat(),
                    'data_points': len(pair_data),
                    'data': pair_data
                }
                
                # Convertir a bytes
                file_buffer = io.BytesIO()
                file_buffer.write(json.dumps(file_content, indent=2).encode())
                file_buffer.seek(0)
                
                # Metadata del archivo
                file_metadata = {
                    'name': f'{pair}_data_{timestamp}.json',
                    'parents': [self.folder_id] if self.folder_id else []
                }
                
                # Upload
                media = MediaIoBaseUpload(
                    file_buffer,
                    mimetype='application/json',
                    resumable=True
                )
                
                file = self.service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id'
                ).execute()
                
                print(f"Uploaded {pair} data to Drive: {file.get('id')}")
            
            return True
            
        except Exception as e:
            print(f"Error uploading forex data: {e}")
            return False
    
    def upload_signals(self, signals):
        """Subir señales a Google Drive"""
        try:
            if not self.service:
                return False
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Preparar archivo
            file_buffer = io.BytesIO()
            file_buffer.write(json.dumps(signals, indent=2).encode())
            file_buffer.seek(0)
            
            # Metadata
            file_metadata = {
                'name': f'trading_signals_{timestamp}.json',
                'parents': [self.folder_id] if self.folder_id else []
            }
            
            # Upload
            media = MediaIoBaseUpload(
                file_buffer,
                mimetype='application/json',
                resumable=True
            )
            
            # Verificar si existe archivo "latest_signals.json"
            existing_file = self._find_file('latest_signals.json')
            
            if existing_file:
                # Actualizar archivo existente
                updated_file = self.service.files().update(
                    fileId=existing_file['id'],
                    media_body=media
                ).execute()
                print(f"Updated latest signals in Drive")
            else:
                # Crear nuevo archivo
                file = self.service.files().create(
                    body={**file_metadata, 'name': 'latest_signals.json'},
                    media_body=media,
                    fields='id'
                ).execute()
                print(f"Created new signals file in Drive")
            
            # También crear archivo con timestamp
            file_with_timestamp = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            return True
            
        except Exception as e:
            print(f"Error uploading signals: {e}")
            return False
    
    def download_pair_data(self, pair):
        """Descargar datos de un par específico"""
        try:
            if not self.service:
                return []
            
            # Buscar archivos del par
            query = f"name contains '{pair}_data_' and parents in '{self.folder_id}'"
            
            results = self.service.files().list(
                q=query,
                orderBy='modifiedTime desc',
                pageSize=1,
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            
            if not files:
                print(f"No data files found for {pair}")
                return []
            
            # Descargar el archivo más reciente
            file_id = files[0]['id']
            
            request = self.service.files().get_media(fileId=file_id)
            file_buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(file_buffer, request)
            
            done = False
            while done is False:
                status, done = downloader.next_chunk()
            
            # Parsear JSON
            file_buffer.seek(0)
            file_content = json.loads(file_buffer.read().decode())
            
            return file_content.get('data', [])
            
        except Exception as e:
            print(f"Error downloading data for {pair}: {e}")
            return []
    
    def _find_file(self, filename):
        """Buscar archivo por nombre"""
        try:
            query = f"name='{filename}'"
            if self.folder_id:
                query += f" and parents in '{self.folder_id}'"
            
            results = self.service.files().list(
                q=query,
                fields="files(id, name)"
            ).execute()
            
            files = results.get('files', [])
            return files[0] if files else None
            
        except Exception as e:
            print(f"Error finding file {filename}: {e}")
            return None
    
    def cleanup_old_files(self, days_old=7):
        """Limpiar archivos antiguos"""
        try:
            if not self.service:
                return False
            
            from datetime import timedelta
            cutoff_date = datetime.now() - timedelta(days=days_old)
            cutoff_str = cutoff_date.isoformat()
            
            query = f"modifiedTime < '{cutoff_str}'"
            if self.folder_id:
                query += f" and parents in '{self.folder_id}'"
            
            results = self.service.files().list(q=query).execute()
            files = results.get('files', [])
            
            deleted_count = 0
            for file in files:
                self.service.files().delete(fileId=file['id']).execute()
                deleted_count += 1
            
            print(f"Cleaned up {deleted_count} old files")
            return True
            
        except Exception as e:
            print(f"Error cleaning up files: {e}")
            return False

if __name__ == "__main__":
    manager = GoogleDriveManager()
    
    # Test de funcionalidad
    if manager.service:
        print("Google Drive connection successful")
    else:
        print("Google Drive connection failed")
