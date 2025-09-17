# scripts/update_google_sheets.py
import json
import os
from datetime import datetime
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

class GoogleSheetsUpdater:
    def __init__(self):
        self.spreadsheet_id = os.environ.get('GOOGLE_SPREADSHEET_ID')
        self.service = self._authenticate()
        
    def _authenticate(self):
        """Autenticar con Google Sheets"""
        try:
            credentials_json = os.environ.get('GOOGLE_CREDENTIALS')
            
            if not credentials_json:
                print("Google credentials not found")
                return None
            
            credentials_info = json.loads(credentials_json)
            credentials = Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )
            
            return build('sheets', 'v4', credentials=credentials)
            
        except Exception as e:
            print(f"Error authenticating Google Sheets: {e}")
            return None
    
    def update_signals_sheet(self, signals_data):
        """Actualizar hoja de señales"""
        try:
            if not self.service:
                return False
            
            # Preparar datos para la hoja
            values = [
                ['Timestamp', 'Pair', 'Signal', 'Confidence', 'Position Size', 'Price', 'RSI', 'MACD']
            ]
            
            for pair, signal_data in signals_data['signals'].items():
                values.append([
                    signals_data['timestamp'],
                    pair,
                    signal_data['signal'],
                    signal_data['confidence'],
                    signal_data['position_size'],
                    signal_data['current_price'],
                    signal_data.get('rsi', 'N/A'),
                    signal_data.get('macd', 'N/A')
                ])
            
            # Limpiar hoja y agregar nuevos datos
            range_name = 'Current Signals!A:H'
            
            # Limpiar
            self.service.spreadsheets().values().clear(
                spreadsheetId=self.spreadsheet_id,
                range=range_name
            ).execute()
            
            # Actualizar
            body = {'values': values}
            
            result = self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='USER_ENTERED',
                body=body
            ).execute()
            
            print(f"Updated {result.get('updatedCells')} cells in Google Sheets")
            return True
            
        except Exception as e:
            print(f"Error updating signals sheet: {e}")
            return False
    
    def append_to_history(self, signals_data):
        """Agregar señales al historial"""
        try:
            if not self.service:
                return False
            
            values = []
            
            for pair, signal_data in signals_data['signals'].items():
                values.append([
                    signals_data['timestamp'],
                    pair,
                    signal_data['signal'],
                    signal_data['confidence'],
                    signal_data['position_size'],
                    signal_data['current_price'],
                    'Open',  # Status
                    '',      # Result (to be filled later)
                    ''       # PnL (to be filled later)
                ])
            
            if values:
                range_name = 'Trading History!A:I'
                body = {'values': values}
                
                result = self.service.spreadsheets().values().append(
                    spreadsheetId=self.spreadsheet_id,
                    range=range_name,
                    valueInputOption='USER_ENTERED',
                    insertDataOption='INSERT_ROWS',
                    body=body
                ).execute()
                
                print(f"Appended {len(values)} rows to history")
                return True
            
        except Exception as e:
            print(f"Error appending to history: {e}")
            return False
    
    def update_dashboard_metrics(self, signals_data):
        """Actualizar métricas del dashboard"""
        try:
            if not self.service:
                return False
            
            signals = signals_data['signals']
            
            # Calcular métricas
            total_signals = len(signals)
            buy_signals = len([s for s in signals.values() if s['signal'] == 'BUY'])
            sell_signals = len([s for s in signals.values() if s['signal'] == 'SELL'])
            avg_confidence = sum(s['confidence'] for s in signals.values()) / total_signals if total_signals > 0 else 0
            high_conf_signals = len([s for s in signals.values() if s['confidence'] > 0.8])
            
            # Preparar actualizaciones
            updates = [
                {
                    'range': 'Dashboard!B2',
                    'values': [[datetime.now().strftime('%Y-%m-%d %H:%M:%S')]]
                },
                {
                    'range': 'Dashboard!B3',
                    'values': [[total_signals]]
                },
                {
                    'range': 'Dashboard!B4',
                    'values': [[buy_signals]]
                },
                {
                    'range': 'Dashboard!B5',
                    'values': [[sell_signals]]
                },
                {
                    'range': 'Dashboard!B6',
                    'values': [[round(avg_confidence, 3)]]
                },
                {
                    'range': 'Dashboard!B7',
                    'values': [[high_conf_signals]]
                }
            ]
            
            # Ejecutar actualizaciones batch
            body = {'valueInputOption': 'USER_ENTERED', 'data': updates}
            
            result = self.service.spreadsheets().values().batchUpdate(
                spreadsheetId=self.spreadsheet_id,
                body=body
            ).execute()
            
            print("Dashboard metrics updated")
            return True
            
        except Exception as e:
            print(f"Error updating dashboard: {e}")
            return False
    
    def create_sheets_if_not_exist(self):
        """Crear hojas si no existen"""
        try:
            if not self.service:
                return False
            
            # Obtener hojas existentes
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()
            
            existing_sheets = [sheet['properties']['title'] for sheet in spreadsheet['sheets']]
            
            # Hojas requeridas
            required_sheets = ['Current Signals', 'Trading History', 'Dashboard']
            
            requests = []
            
            for sheet_name in required_sheets:
                if sheet_name not in existing_sheets:
                    requests.append({
                        'addSheet': {
                            'properties': {
                                'title': sheet_name
                            }
                        }
                    })
            
            if requests:
                body = {'requests': requests}
                
                self.service.spreadsheets().batchUpdate(
                    spreadsheetId=self.spreadsheet_id,
                    body=body
                ).execute()
                
                print(f"Created {len(requests)} new sheets")
            
            return True
            
        except Exception as e:
            print(f"Error creating sheets: {e}")
            return False

def main():
    """Función principal para GitHub Actions"""
    try:
        updater = GoogleSheetsUpdater()
        
        # Cargar señales
        with open('data/latest_signals.json', 'r') as f:
            signals_data = json.load(f)
        
        # Crear hojas si no existen
        updater.create_sheets_if_not_exist()
        
        # Actualizar datos
        updater.update_signals_sheet(signals_data)
        updater.append_to_history(signals_data)
        updater.update_dashboard_metrics(signals_data)
        
        print("Google Sheets update completed successfully")
        return True
        
    except Exception as e:
        print(f"Error in Google Sheets update: {e}")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
